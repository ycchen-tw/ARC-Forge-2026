"""Block-FP8 (W8A8) QAT of the dense projections the deployed gEAN320 INT3 model keeps in FP8.

The deployed model runs 144 projections (12 attention layers x q/k/v, 36 GDN layers x in_proj_qkv/in_proj_z/out_proj) as
SGLang ``Fp8LinearMethod(activation_scheme="dynamic", weight_block_size=[128, 128])``: the weight is e4m3 with one fp32
scale per 128x128 block (``work/qwen38fn_eval/quantize_nonmoe_fp8.py``), the activation is quantized to e4m3 per token
and 128 channels at run time, the GEMM accumulates in fp32 with both scales applied per K block. The trainer's LoRA wraps
the BF16 master of those weights, so without this module its forward is the *unquantized* model and the LoRA delta never
meets the grid it will be rounded onto at serving time.

``--dense-qat fp8`` replaces the forward of those 144 LoRA modules by ``DenseFP8QAT``:

    w_eff = W.float() + scaling * (B.float() @ A.float())     # the merged weight, fp32
    q, sw = quantize_block_fp8(w_eff)                          # exactly the deployed weight quantizer
    xq, sx = quantize(x)                                       # the MoE path's SGLang 1x128 activation quantizer
    y = block-FP8 GEMM(xq, sx, q, sw) -> bf16                  # SGLang fused_moe_kernel, one expert

Backward is a straight-through estimator: dX = dY @ dequant(q, sw) (the weight the forward used), and the LoRA gradients
are the standard ones with the *quantized* input, ``dB = s * dY^T (x_dq A^T)``, ``dA = s * (dY B)^T x_dq``, in fp32 and
chunked over rows. W gets no gradient and no [N, K] gradient is formed.

The merged export (merged_export.py v2) writes ``quantize_block_fp8(effective_weight(...))`` -- the same functions on the
same device -- so the exported e4m3/scale_inv are bit-for-bit what the forward multiplied with.

The quantization functions here are plain torch and import without Triton; the GEMM and the activation quantizer load the
pinned SGLang kernels lazily (upstream_fp8.py).

``--dense-qat-gemm cutlass`` (set_gemm) runs the forward GEMM on flashinfer's SM100 CUTLASS groupwise-FP8 kernel instead
of the Triton fused_moe_kernel: the same quantized operands and the same per-128-K-block scaling with fp32 accumulation
(bit for bit equal to the Triton GEMM on all six projection shapes at 180k rows, check_dense_qat_cutlass.py -- measured,
not promised by either library; ~1.7x faster). flashinfer comes from the pinned PYTHONPATH overlay of
flashinfer_overlay.py (venv-fn-train has none) and JIT-compiles with `ninja` (the venv's bin/ on PATH) and nvcc
($CUDA_HOME) into $FLASHINFER_WORKSPACE_BASE.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

BLOCK = 128
FP8_MAX = 448.0
ADAPTER = 'default'
# Rows per step of the LoRA gradient: bounds the fp32 copies of dY ([rows, N]) and of the dequantized x ([rows, K]).
GRAD_ROWS = 16384
# 'lora_grad' (trainer_kernels --fast-kernels lora_grad): the LoRA gradients through lora_grad.py on the e4m3 input
GRAD_KERNEL = {'kernel': 'fp32'}
# The forward GEMM (--dense-qat-gemm, set_gemm): 'triton' = block_fp8_gemm, 'cutlass' = cutlass_block_fp8_gemm
GEMM_KERNELS = ('triton', 'cutlass')
GEMM = {'kernel': 'triton'}
COUNTERS = {'forward': 0, 'backward': 0, 'cutlass_forward': 0}
# Debug (train_tito_mm.py --dense-qat-check-export): None, or {checkpoint key: (sha256 of q, sha256 of sw)} of the latest
# forward of each module, for merged_export to compare against the tensors it writes.
CAPTURE = None
INSTALLED = {}  # checkpoint module name -> PEFT module, set by install()
_SEEN = {'forward': 0, 'backward': 0, 'cutlass_forward': 0}


def quantize_block_fp8(w):
    """Symmetric per-[128,128]-block e4m3 of an fp32 [N, K] weight -> (q float8_e4m3fn [N, K], scale fp32 [N/128, K/128]).

    The operations, their order and dtypes are those of ``quantize_nonmoe_fp8.quantize`` (which made the deployed
    weights); a fully-zero block gets scale 1. Run it on the same device kind as the forward (CUDA) for bitwise results.
    """
    if w.dtype != torch.float32 or w.dim() != 2 or w.shape[0] % BLOCK or w.shape[1] % BLOCK:
        raise ValueError(f'block FP8 needs an fp32 [N%128, K%128] weight, got {w.dtype} {tuple(w.shape)}')
    out_dim, in_dim = w.shape
    blocked = w.view(out_dim // BLOCK, BLOCK, in_dim // BLOCK, BLOCK)
    amax = blocked.abs().amax(dim=(1, 3))
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = blocked / scale[:, None, :, None]
    q = q.clamp(-FP8_MAX, FP8_MAX).view(out_dim, in_dim).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


def dequantize_block_fp8(q, scale, dtype):
    n, k = q.shape
    return (q.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK) * scale[:, None, :, None]).view(n, k).to(dtype)


def effective_weight(w, a, b, scaling):
    """The LoRA-merged weight in fp32, ``W + scaling * (B @ A)``; the one expression the forward and the export share."""
    return w.float() + scaling * (b.float() @ a.float())


def quantize_activation(x):
    """[M, K] bf16 -> (e4m3 [M, K], fp32 scale [M, K/128]): SGLang's per-token-group 1x128 dynamic quantizer."""
    from fp8_moe import quantize

    return quantize(x)


def rows_per_call(k, n):
    """Most rows one quantize_activation / block_fp8_gemm call takes for a [*, K] input and an [N, K] weight: their
    Triton kernels index with int32, and rows x max(K, N) elements must stay below 2**31 (a multiple of the GEMM's
    64-row tile). At 2560 -> 12288 (attention q_proj) that is 174,720 rows, less than one 200k-token bin."""
    return (2**31 - 1) // max(k, n) // 64 * 64


def dequantize_activation(xq, sx, dtype=torch.float32):
    m, k = xq.shape
    return (xq.float().view(m, k // BLOCK, BLOCK) * sx[:, :, None]).view(m, k).to(dtype)


def block_fp8_gemm(xq, sx, q, sw, out=None):
    """y[M, N] bf16 = sum over K blocks of (xq_blk @ q_blk^T) * sx[:, blk] * sw[:, blk], fp32 accumulation.

    SGLang's unmodified ``fused_moe_kernel`` (block-quant FP8 W8A8 path, the one fp8_moe.py runs the experts with) with a
    single expert and the identity routing. ``out``: a contiguous bf16 [M, N] to write into (a row slice of a bigger
    output), else a new one. The kernel's offsets are int32: M x max(K, N) must stay below 2**31 (rows_per_call)."""
    import triton
    import triton.language as tl
    from upstream_fp8 import kernels

    m, k = xq.shape
    n = q.shape[0]
    if q.shape[1] != k or sx.shape != (m, k // BLOCK) or sw.shape != (n // BLOCK, k // BLOCK) or xq.stride() != (k, 1):
        raise ValueError(f'block FP8 GEMM shapes: x {tuple(xq.shape)} sx {tuple(sx.shape)} w {tuple(q.shape)} '
                         f'sw {tuple(sw.shape)}')
    if q.dtype != torch.float8_e4m3fn or xq.dtype != torch.float8_e4m3fn or not q.is_contiguous() \
            or not sw.is_contiguous() or sw.dtype != torch.float32 or sx.dtype != torch.float32:
        raise ValueError('block FP8 GEMM wants contiguous e4m3 operands with fp32 scales')
    if m * max(k, n) >= 2**31:
        raise ValueError(f'block FP8 GEMM: {m} rows x max(K {k}, N {n}) overflow its int32 offsets; '
                         f'call it on at most rows_per_call(K, N) = {rows_per_call(k, n)} rows')
    if out is not None and (out.shape != (m, n) or out.dtype != torch.bfloat16 or not out.is_contiguous()):
        raise ValueError(f'block FP8 GEMM out: {tuple(out.shape)} {out.dtype} contiguous={out.is_contiguous()}, '
                         f'expected contiguous bf16 {(m, n)}')
    blocks = triton.cdiv(m, 64)
    ids = torch.arange(blocks * 64, dtype=torch.int32, device=xq.device)  # ids >= m are padding rows (masked)
    experts = torch.zeros(blocks, dtype=torch.int32, device=xq.device)
    # torch.full, not torch.tensor([...]): a host list is a pageable H2D copy that synchronizes the stream every call
    count = torch.full((1,), blocks * 64, dtype=torch.int32, device=xq.device)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    weight, scale = q.unsqueeze(0), sw.unsqueeze(0)
    gemm, _ = kernels()
    gemm[(blocks * triton.cdiv(n, 128),)](
        a_ptr=xq, a_desc=None, b_ptr=weight, b_desc=None, bias_ptr=None, c_ptr=out,
        a_scale_ptr=sx, b_scale_ptr=scale, topk_weights_ptr=None, sorted_token_ids_ptr=ids,
        expert_ids_ptr=experts, num_tokens_post_padded_ptr=count, add_mask_ptr=None,
        N=n, K=k, EM=ids.numel(), num_valid_tokens=m,
        stride_am=xq.stride(0), stride_ak=1,
        stride_be=weight.stride(0), stride_bk=weight.stride(2), stride_bn=weight.stride(1),
        stride_bias_e=0, stride_bias_n=0, stride_cm=out.stride(0), stride_cn=1,
        stride_asm=sx.stride(0), stride_ask=1,
        stride_bse=scale.stride(0), stride_bsk=scale.stride(2), stride_bsn=scale.stride(1),
        group_n=BLOCK, group_k=BLOCK, BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=8,
        MUL_ROUTED_WEIGHT=False, top_k=1, compute_type=tl.bfloat16, use_fp8_w8a8=True, use_int8_w8a8=False,
        use_int8_w8a16=False, per_channel_quant=False, even_Ks=True, c_sorted=False, filter_expert=False,
        swap_ab=False, FUSE_ADD_TO_OUTPUT=False, MASK_OUTPUT=False, FUSE_SUM_ALL_REDUCE=False,
        LORA_PRESERVE_BASE=False, ROUTER_TOPK=1, FUSE_SWIGLU=False, num_warps=4, num_stages=3)
    return out


def set_gemm(kernel, device=None):
    """Select the forward GEMM. 'cutlass' checks that the imported flashinfer is the pinned overlay, that
    $FLASHINFER_WORKSPACE_BASE names the JIT cache (the serving venv's flashinfer builds the same module names against
    another tvm-ffi into ~/.cache by default, so the trainer must not share it), and builds/loads the module on
    ``device`` with one small GEMM so the first training step does not wait for nvcc."""
    import os
    import shutil

    if kernel not in GEMM_KERNELS:
        raise ValueError(f'dense QAT GEMM {kernel!r}, expected one of {GEMM_KERNELS}')
    if kernel == 'cutlass':
        if not os.environ.get('FLASHINFER_WORKSPACE_BASE'):
            raise RuntimeError('--dense-qat-gemm cutlass needs FLASHINFER_WORKSPACE_BASE (the trainer\'s own flashinfer '
                               'JIT cache) set before flashinfer is imported')
        if shutil.which('ninja') is None:
            raise RuntimeError('--dense-qat-gemm cutlass: flashinfer JIT-builds with ninja; put the venv bin/ on PATH')
        if device is None or torch.cuda.get_device_capability(device)[0] != 10:
            raise RuntimeError(f'--dense-qat-gemm cutlass is the SM100 kernel; device {device}')
        from flashinfer_overlay import check_imported

        check_imported()
        xq, sx = quantize_activation(torch.ones((70, BLOCK), dtype=torch.bfloat16, device=device))
        q, sw = quantize_block_fp8(torch.ones((4 * BLOCK, BLOCK), dtype=torch.float32, device=device))
        seen = COUNTERS['cutlass_forward']
        if not torch.equal(cutlass_block_fp8_gemm(xq, sx, q, sw), block_fp8_gemm(xq, sx, q, sw)):
            raise RuntimeError('CUTLASS and Triton block-FP8 GEMMs differ on an all-ones product')
        COUNTERS['cutlass_forward'] = seen  # the warm-up is not a training forward
    GEMM['kernel'] = kernel


def cutlass_block_fp8_gemm(xq, sx, q, sw):
    """block_fp8_gemm's product on flashinfer's SM100 CUTLASS groupwise kernel (1x128 activation / 128x128 weight
    scales, fp32 accumulation, BF16 out). The kernel reads the activation scales MN-major with a row stride of M, which
    must be a multiple of 4 floats: the rows past the last multiple of 4 (at most 3) run as a 4-row zero-padded call."""
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    m, k = xq.shape
    n = q.shape[0]
    if q.shape[1] != k or sx.shape != (m, k // BLOCK) or sw.shape != (n // BLOCK, k // BLOCK) or xq.stride() != (k, 1):
        raise ValueError(f'block FP8 GEMM shapes: x {tuple(xq.shape)} sx {tuple(sx.shape)} w {tuple(q.shape)} '
                         f'sw {tuple(sw.shape)}')
    if q.dtype != torch.float8_e4m3fn or xq.dtype != torch.float8_e4m3fn or not q.is_contiguous() \
            or sw.dtype != torch.float32 or sx.dtype != torch.float32 or (n // BLOCK) % 4:
        raise ValueError('CUTLASS block FP8 GEMM wants contiguous e4m3 operands, fp32 scales and N/128 % 4 == 0')
    y = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    sw_mn = sw.t().contiguous()
    head = m - m % 4
    if head:
        gemm_fp8_nt_groupwise(xq[:head], q, sx[:head].t().contiguous(), sw_mn, scale_major_mode='MN', mma_sm=2,
                              out=y[:head], backend='cutlass')
    if head < m:
        tail = m - head
        xt = torch.zeros((4, k), dtype=xq.dtype, device=xq.device)
        xt[:tail] = xq[head:]
        st = torch.ones((k // BLOCK, 4), dtype=torch.float32, device=xq.device)
        st[:, :tail] = sx[head:].t()
        yt = gemm_fp8_nt_groupwise(xt, q, st, sw_mn, scale_major_mode='MN', mma_sm=1,
                                   out=torch.empty((4, n), dtype=torch.bfloat16, device=xq.device), backend='cutlass')
        y[head:] = yt[:tail]
    COUNTERS['cutlass_forward'] += 1
    return y


def _digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


class DenseFP8QAT(torch.autograd.Function):
    """y = block-FP8 GEMM of the quantized input and the quantized LoRA-merged weight; STE backward (module docstring)."""

    @staticmethod
    def forward(ctx, x, w, a, b, scaling, tag):
        if x.dim() != 2 or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or a.dtype != torch.float32 \
                or b.dtype != torch.float32 or w.requires_grad:
            raise TypeError(f'{tag}: x {x.dtype} {tuple(x.shape)}, W {w.dtype} grad={w.requires_grad}, '
                            f'A {a.dtype}, B {b.dtype}')
        q, sw = quantize_block_fp8(effective_weight(w, a, b, scaling))
        rows = rows_per_call(x.shape[1], w.shape[0])
        if GEMM['kernel'] == 'cutlass':
            # the quantizer in the same row chunks as below (row-local, so the same bits), then one CUTLASS GEMM
            chunks = [quantize_activation(x[lo:lo + rows]) for lo in range(0, x.shape[0], rows)]
            xq, sx = chunks[0] if len(chunks) == 1 else (torch.cat([c[0] for c in chunks]),
                                                         torch.cat([c[1] for c in chunks]))
            del chunks
            y = cutlass_block_fp8_gemm(xq, sx, q, sw)
        elif x.shape[0] <= rows:
            xq, sx = quantize_activation(x)
            y = block_fp8_gemm(xq, sx, q, sw)
        else:
            # A packed bin longer than the kernels' int32 reach (rows_per_call): quantize and multiply in row chunks.
            # Both are row-local (one scale per row and 128 channels; each output row is its own K-ordered dot
            # products), so every row is bit for bit what a call on fewer rows gives (test_fp8_dense_qat.py).
            chunks = [quantize_activation(x[lo:lo + rows]) for lo in range(0, x.shape[0], rows)]
            xq = torch.cat([c[0] for c in chunks])
            sx = torch.cat([c[1] for c in chunks])
            del chunks
            y = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
            for lo in range(0, x.shape[0], rows):
                block_fp8_gemm(xq[lo:lo + rows], sx[lo:lo + rows], q, sw, out=y[lo:lo + rows])
        if CAPTURE is not None:
            CAPTURE[tag] = (_digest(q), _digest(sw))
        ctx.save_for_backward(xq, sx, q, sw, a, b)
        ctx.scaling = scaling
        COUNTERS['forward'] += 1
        return y

    @staticmethod
    def backward(ctx, dy):
        xq, sx, q, sw, a, b = ctx.saved_tensors
        dx = da = db = None
        dy = dy.contiguous()
        if dy.dtype != torch.bfloat16:
            raise TypeError(f'output gradient is {dy.dtype}, expected bfloat16')
        if ctx.needs_input_grad[0]:
            dx = dy @ dequantize_block_fp8(q, sw, torch.bfloat16)
        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            if not (ctx.needs_input_grad[2] and ctx.needs_input_grad[3]):
                raise RuntimeError('LoRA A and B must both train')
            if GRAD_KERNEL['kernel'] == 'lora_grad':
                da, db = _lora_grads_triton(dy, xq, sx, a, b, ctx.scaling)
                COUNTERS['backward'] += 1
                return dx, None, da, db, None, None
            if GRAD_KERNEL['kernel'] != 'fp32':
                raise ValueError(f"GRAD_KERNEL {GRAD_KERNEL['kernel']!r}: 'fp32' or 'lora_grad'")
            da = torch.zeros_like(a)
            db = torch.zeros_like(b)
            for lo in range(0, dy.shape[0], GRAD_ROWS):
                g = dy[lo:lo + GRAD_ROWS].float()
                x = dequantize_activation(xq[lo:lo + GRAD_ROWS], sx[lo:lo + GRAD_ROWS])
                db += g.T @ (x @ a.T)
                da += (g @ b).T @ x
            da *= ctx.scaling
            db *= ctx.scaling
        COUNTERS['backward'] += 1
        return dx, None, da, db, None, None


def _lora_grads_triton(dy, xq, sx, a, b, scaling):
    """The same dA / dB without fp32 copies: s = x_dq A^T and dA from the e4m3 input and its scales, ds = dY B on BF16
    operands (B as hi + lo, the reduction is over N), dB and dA reduced over T by lora_grad.tgrad."""
    import lora_grad
    r = a.shape[0]
    s = lora_grad.fp8_rows_times(xq, sx, a)
    hi = b.to(torch.bfloat16)
    d2 = torch.mm(dy, torch.cat([hi, (b - hi.float()).to(torch.bfloat16)], dim=1), out_dtype=torch.float32)
    ds = d2[:, :r] + d2[:, r:]
    return lora_grad.tgrad(xq, ds, sx).t() * scaling, lora_grad.tgrad(dy, s) * scaling


def qat_forward(self, x):
    """Replacement ``forward`` of a PEFT ``lora.Linear`` (bound on the instance by install())."""
    base = self.base_layer
    shape = x.shape
    y = DenseFP8QAT.apply(x.reshape(-1, shape[-1]).contiguous(), base.weight, self.lora_A[ADAPTER].weight,
                          self.lora_B[ADAPTER].weight, self.scaling[ADAPTER], self._qat_key)
    return y.view(*shape[:-1], y.shape[-1])


def add_arguments(parser, debug=False):
    """``--dense-qat`` / ``--dense-qat-reference`` (and, with ``debug``, ``--dense-qat-check-export``) of the trainers."""
    parser.add_argument('--dense-qat', choices=['none', 'fp8'], default='none',
                        help='fp8: the LoRA projections the deployed model runs as block FP8 (listed by '
                             '--dense-qat-reference) are trained as quantize(x) @ quantize(W + LoRA)^T with a '
                             'straight-through backward, and the merged export is fn-lora-merged-v2 '
                             '(fp8_dense_qat.py, merged_export.py)')
    parser.add_argument('--dense-qat-reference',
                        help='the deployed package directory (config.json + safetensors index): names the FP8 '
                             'projections, and at start the trainer raises unless quantizing its BF16 base gives '
                             "that package's FP8 weights and scale_inv bit for bit")
    parser.add_argument('--dense-qat-gemm', choices=list(GEMM_KERNELS), default='triton',
                        help='--dense-qat fp8 forward GEMM: triton = SGLang fused_moe_kernel (block_fp8_gemm); cutlass = '
                             'flashinfer\'s SM100 CUTLASS groupwise kernel (needs the flashinfer_overlay.py overlay on '
                             'PYTHONPATH, FLASHINFER_WORKSPACE_BASE and ninja on PATH; same operands, measured bit for '
                             'bit equal to triton)')
    if debug:
        parser.add_argument('--dense-qat-check-export', action='store_true',
                            help='debug: before each merged export run one more no-grad forward and compare the '
                                 'exported e4m3/scale_inv bitwise with the ones that forward quantized with '
                                 '(needs --save-merged-every; the capture hashes every module, so it is slow)')


def check_arguments(parser, args, debug=False):
    if (args.dense_qat == 'fp8') != bool(args.dense_qat_reference):
        parser.error('--dense-qat fp8 goes with --dense-qat-reference <deployed package>, and only with it')
    if debug and args.dense_qat_check_export and args.dense_qat != 'fp8':
        parser.error('--dense-qat-check-export needs --dense-qat fp8')
    if args.dense_qat_gemm != 'triton' and args.dense_qat != 'fp8':
        parser.error('--dense-qat-gemm cutlass needs --dense-qat fp8')


def reference_modules(package):
    """The projections the deployed package stores as block FP8: ``quantization_config.arc_int3.dense_fp8_modules``."""
    config = json.loads((Path(package) / 'config.json').read_text())
    modules = config['quantization_config']['arc_int3']['dense_fp8_modules']
    if not modules or len(set(modules)) != len(modules):
        raise RuntimeError(f'{package}: empty or repeated dense_fp8_modules')
    return list(modules)


def verify_against_package(weights, package, device):
    """Raise unless ``quantize_block_fp8(weights[m])`` equals the package's ``m.weight`` / ``m.weight_scale_inv`` bit for
    bit, for every module m in ``weights`` ({module name: BF16 [N, K] tensor}). Returns {'modules', 'weight_elements',
    'scale_elements'} of what was compared."""
    from safetensors import safe_open

    package = Path(package)
    index = json.loads((package / 'model.safetensors.index.json').read_text())['weight_map']
    by_shard = defaultdict(list)
    for module in weights:
        for suffix in ('weight', 'weight_scale_inv'):
            if f'{module}.{suffix}' not in index:
                raise KeyError(f'{package}: no {module}.{suffix} in the weight map')
        by_shard[index[f'{module}.weight']].append(module)
    failures, elements, scales = [], 0, 0
    for shard, modules in sorted(by_shard.items()):
        with safe_open(str(package / shard), framework='pt') as handle:
            for module in modules:
                w = weights[module]
                if w.dtype != torch.bfloat16:
                    raise TypeError(f'{module}: base weight is {w.dtype}; the QAT master must be BF16')
                q, sw = quantize_block_fp8(w.to(device=device, dtype=torch.float32))
                q_ref = handle.get_tensor(f'{module}.weight')
                sw_ref = handle.get_tensor(f'{module}.weight_scale_inv')
                if q_ref.dtype != torch.float8_e4m3fn or sw_ref.dtype != torch.float32 or q_ref.shape != q.shape \
                        or sw_ref.shape != sw.shape:
                    raise TypeError(f'{package}: {module} is {q_ref.dtype} {tuple(q_ref.shape)} / {sw_ref.dtype} '
                                    f'{tuple(sw_ref.shape)}, expected e4m3 {tuple(q.shape)} / fp32 {tuple(sw.shape)}')
                bad_q = int((q.view(torch.uint8).cpu() != q_ref.view(torch.uint8)).sum())
                bad_s = int((sw.cpu().view(torch.int32) != sw_ref.view(torch.int32)).sum())
                elements += q.numel()
                scales += sw.numel()
                if bad_q or bad_s:
                    failures.append(f'{module}: {bad_q} of {q.numel()} e4m3 bytes, {bad_s} of {sw.numel()} scales differ')
    if failures:
        raise RuntimeError(f'the trainer\'s block-FP8 quantization of its base differs from {package} in '
                           f'{len(failures)} of {len(weights)} modules (base or quantizer differs from the deployed '
                           f'one): {failures[:5]}')
    return {'modules': len(weights), 'weight_elements': elements, 'scale_elements': scales}


def install(model, package, device, scaling):
    """Run the package's block-FP8 projections of PEFT ``model`` through DenseFP8QAT.

    Call after the LoRA is attached and before FSDP shards the base (the BF16 weights are verified against the package
    here, on ``device``). Binds ``forward`` on each ``lora.Linear`` instance (a class attribute would not win over
    PEFT's), checks the module is the plain single-adapter LoRA this math assumes, and returns
    ``{'modules': {checkpoint module name: PEFT module}, 'verified': {...}}``. ``scaling`` is alpha / rank."""
    from multimodal import adapter_checkpoint_key
    from peft.tuners.lora.layer import Linear as LoraLinear

    wanted = reference_modules(package)
    lora = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            lora[adapter_checkpoint_key(name + '.lora_A.weight').removesuffix('.weight')] = module
    missing = [m for m in wanted if m not in lora]
    if missing:
        raise KeyError(f'{len(missing)} of the package\'s FP8 projections carry no LoRA: {missing[:3]}')
    modules = {m: lora[m] for m in wanted}
    for name, module in modules.items():
        base = module.base_layer
        a, b = module.lora_A[ADAPTER].weight, module.lora_B[ADAPTER].weight
        if (type(base) is not torch.nn.Linear or base.bias is not None or list(module.lora_A) != [ADAPTER]
                or module.active_adapters != [ADAPTER] or module.scaling[ADAPTER] != scaling or module.merged
                or module.disable_adapters or module.use_dora.get(ADAPTER, False)
                or not isinstance(module.lora_dropout[ADAPTER], torch.nn.Identity)
                or a.dtype != torch.float32 or b.dtype != torch.float32 or base.weight.dtype != torch.bfloat16
                or (b.shape[0], a.shape[1]) != tuple(base.weight.shape) or 'forward' in module.__dict__):
            raise RuntimeError(f'{name}: not a plain bias-free single-adapter LoRA over a BF16 Linear with '
                               f'scaling {scaling}')
    verified = verify_against_package({m: module.base_layer.weight.data for m, module in modules.items()}, package, device)
    from types import MethodType

    for name, module in modules.items():
        module._qat_key = name + '.weight'
        module.forward = MethodType(qat_forward, module)
    INSTALLED.clear()
    INSTALLED.update(modules)
    _SEEN.update(COUNTERS)  # assert_step_calls counts from here
    return {'modules': modules, 'verified': verified}


def assert_step_calls(update):
    """After one forward (and backward) of the model: every installed module ran DenseFP8QAT once per forward -- twice
    with the checkpoint recompute when training -- and once in backward. Nothing to check when QAT is off."""
    if not INSTALLED:
        return
    forwards, backwards = COUNTERS['forward'] - _SEEN['forward'], COUNTERS['backward'] - _SEEN['backward']
    cutlass = COUNTERS['cutlass_forward'] - _SEEN['cutlass_forward']
    _SEEN.update(COUNTERS)
    expected = len(INSTALLED) * (2 if update else 1)
    if forwards != expected or backwards != (len(INSTALLED) if update else 0):
        raise RuntimeError(f'dense QAT ran {forwards} forwards / {backwards} backwards this step; expected {expected} / '
                           f'{len(INSTALLED) if update else 0} (a LoRA forward bypassed the QAT path)')
    if cutlass != (forwards if GEMM['kernel'] == 'cutlass' else 0):
        raise RuntimeError(f'{cutlass} of {forwards} dense QAT forwards ran the CUTLASS GEMM; --dense-qat-gemm '
                           f'{GEMM["kernel"]}')

