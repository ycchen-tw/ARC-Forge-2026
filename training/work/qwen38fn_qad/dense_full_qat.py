"""Full-weight block-FP8 QAT of the 144 dense projections the gEAN320 package stores as FP8 (QAD v2).

work/qwen38fn_lora/fp8_dense_qat.py trains a LoRA over a frozen BF16 master. Here the master itself trains: FSDP keeps
it in FP32 and hands the forward its BF16 cast, which is what gets block-quantized:

    q, sw = quantize_block_fp8(W_bf16.float())          # the deployed weight quantizer (fp8_dense_qat), on the GPU
    y     = block-FP8 GEMM(quantize(x), q, sw) -> bf16   # SGLang fused_moe_kernel, one expert

so the deployed weight is Q(bf16(master)) and the exporter writes exactly that (merged_export-style bitwise check in
probes). Backward is straight-through on both quantizers: dX = dY @ dequant(q, sw), dW = dY^T @ dequant(x_q) in FP32
(TF32 off; the operands are BF16 / E4M3-dequantized values, exact in FP32), chunked over rows, returned in the
param's compute dtype for FSDP to reduce in FP32.
"""

import sys
from pathlib import Path

import torch

from qad_timing import section
from scale_grad_triton import dense_weight_grad, dense_weight_grad_quack

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen38fn_lora"))
import fp8_dense_qat as F  # noqa: E402

COUNTERS = {"forward": 0, "backward": 0}
# dW kernel: "fp32" = chunked FP32 GEMM (TF32 off; the reference), "fused" = scale_grad_triton.dense_weight_grad
# (bf16x2 or bf16 operands per "precision", FP32 register accumulation, deterministic), "quack" =
# scale_grad_triton.dense_weight_grad_quack (pre-pass + quack SM100 GEMM, same rounding). Set by the trainer.
DENSE_GRAD = {"mode": "fp32", "precision": "bf16x2"}
GRAD_ROWS = 16384
INSTALLED = {}
_SEEN = {"forward": 0, "backward": 0, "cutlass": 0}


def _weight_grad(dy, xq, sx, shape):
    """dW = dY^T @ dequant(x_q) in FP32 (TF32 off), chunked over rows."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        dw = torch.zeros(shape, dtype=torch.float32, device=dy.device)
        for lo in range(0, dy.shape[0], GRAD_ROWS):
            dw += dy[lo:lo + GRAD_ROWS].float().T @ F.dequantize_activation(xq[lo:lo + GRAD_ROWS], sx[lo:lo + GRAD_ROWS])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return dw


class DenseFP8FullQAT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, tag):
        if x.dim() != 2 or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            raise TypeError(f"{tag}: x {x.dtype} {tuple(x.shape)}, W {w.dtype} (FSDP must hand the forward BF16)")
        q, sw = F.quantize_block_fp8(w.float())
        # A packed bin longer than the Triton kernels' int32 reach (F.rows_per_call: 174,720 rows at 2560 -> 12288, less
        # than one 200k-token bin) is quantized, and on Triton multiplied, in row chunks, as fp8_dense_qat.DenseFP8QAT
        # does: both are row-local, so every row is bit for bit what one call gives.
        rows = F.rows_per_call(x.shape[1], w.shape[0])
        if x.shape[0] <= rows:
            xq, sx = F.quantize_activation(x)
        else:
            chunks = [F.quantize_activation(x[lo:lo + rows]) for lo in range(0, x.shape[0], rows)]
            xq, sx = torch.cat([c[0] for c in chunks]), torch.cat([c[1] for c in chunks])
            del chunks
        # --dense-qat-gemm (F.set_gemm): the same operands on flashinfer's CUTLASS kernel, bitwise equal to the Triton one
        if F.GEMM["kernel"] == "cutlass":
            y = F.cutlass_block_fp8_gemm(xq, sx, q, sw)
        elif x.shape[0] <= rows:
            y = F.block_fp8_gemm(xq, sx, q, sw)
        else:
            y = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
            for lo in range(0, x.shape[0], rows):
                F.block_fp8_gemm(xq[lo:lo + rows], sx[lo:lo + rows], q, sw, out=y[lo:lo + rows])
        ctx.save_for_backward(xq, sx, q, sw)
        ctx.w_dtype = w.dtype
        COUNTERS["forward"] += 1
        return y

    @staticmethod
    def backward(ctx, dy):
        xq, sx, q, sw = ctx.saved_tensors
        dy = dy.contiguous()
        if dy.dtype != torch.bfloat16:
            raise TypeError(f"output gradient is {dy.dtype}, expected bfloat16")
        dx = dy @ F.dequantize_block_fp8(q, sw, torch.bfloat16) if ctx.needs_input_grad[0] else None
        dw = None
        if ctx.needs_input_grad[1]:
            with section("dense_dw"):
                if DENSE_GRAD["mode"] == "fused":
                    dw = dense_weight_grad(dy, xq, sx, precision=DENSE_GRAD["precision"])
                elif DENSE_GRAD["mode"] == "quack":
                    dw = dense_weight_grad_quack(dy, xq, sx, precision=DENSE_GRAD["precision"])
                elif DENSE_GRAD["mode"] == "fp32":
                    dw = _weight_grad(dy, xq, sx, q.shape)
                else:
                    raise ValueError(f"dense gradient mode {DENSE_GRAD['mode']!r}")
                dw = dw.to(ctx.w_dtype)
        COUNTERS["backward"] += 1
        return dx, dw, None


def _forward(self, x):
    shape = x.shape
    y = DenseFP8FullQAT.apply(x.reshape(-1, shape[-1]).contiguous(), self.weight, self._qad_key)
    return y.view(*shape[:-1], y.shape[-1])


def install(model, package, device):
    """Bind the 144 package FP8 projections of ``model`` (Qwen4ExpForConditionalGeneration, no PEFT) to
    DenseFP8FullQAT, after checking that quantizing their BF16 weights gives the package's FP8 bit for bit.
    Call before FSDP. Returns {checkpoint module name: nn.Linear}."""
    from types import MethodType

    wanted = F.reference_modules(package)
    named = dict(model.named_modules())
    modules = {}
    for key in wanted:  # checkpoint names are model.language_model.layers.*, the same as in this model
        module = named.get(key)
        if type(module) is not torch.nn.Linear or module.bias is not None or module.weight.dtype != torch.bfloat16:
            raise RuntimeError(f"{key}: expected a bias-free BF16 nn.Linear, found {type(module).__name__}")
        modules[key] = module
    verified = F.verify_against_package({k: m.weight.data for k, m in modules.items()}, package, device)
    for key, module in modules.items():
        module._qad_key = key + ".weight"
        module.forward = MethodType(_forward, module)
    INSTALLED.clear()
    INSTALLED.update(modules)
    _SEEN.update(COUNTERS, cutlass=F.COUNTERS["cutlass_forward"])
    return modules, verified


def max_rows():
    """The most rows (packed tokens) one forward can take: DenseFP8FullQAT refuses rows x max(in, out) >= 2**31 (the
    activation quantizer's int32 offsets). q_proj 12288 x 2560 gives 174,762 (arc2-hq741 10-30: a 197,732-token bin of
    the 200k prepare.bin_tokens failed the first update)."""
    if not INSTALLED:
        raise RuntimeError("max_rows before install")
    return min((2**31 - 1) // max(m.weight.shape) for m in INSTALLED.values())


def assert_step_calls(update, recompute=True, frozen=0):
    """Every installed projection ran once per forward (twice with the checkpoint recompute) and once in backward.
    ``frozen``: how many of them sit in layers no gradient reaches (the RL trainer's --freeze-first-layers): those run
    one forward and no recompute or backward."""
    forwards, backwards = COUNTERS["forward"] - _SEEN["forward"], COUNTERS["backward"] - _SEEN["backward"]
    cutlass = F.COUNTERS["cutlass_forward"] - _SEEN["cutlass"]
    _SEEN.update(COUNTERS, cutlass=F.COUNTERS["cutlass_forward"])
    if not 0 <= frozen <= len(INSTALLED):
        raise ValueError(f"{frozen} frozen of {len(INSTALLED)} installed projections")
    live = len(INSTALLED) - frozen
    expected_f = live * (2 if update and recompute else 1) + frozen
    expected_b = live if update else 0
    if forwards != expected_f or backwards != expected_b:
        raise RuntimeError(f"dense full QAT ran {forwards} forwards / {backwards} backwards; "
                           f"expected {expected_f} / {expected_b}")
    if cutlass != (forwards if F.GEMM["kernel"] == "cutlass" else 0):
        raise RuntimeError(f"{cutlass} of {forwards} dense full QAT forwards ran the CUTLASS GEMM; "
                           f"--dense-qat-gemm {F.GEMM['kernel']}")
