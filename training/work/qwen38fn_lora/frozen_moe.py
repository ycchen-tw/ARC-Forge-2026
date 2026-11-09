"""Skip unused expert-weight gradients in Liger's frozen-base MoE backward.

Reuse the installed kernels and exact dX/dRouting code. Strict source anchors
make upstream API drift fail at startup. No installed package is edited.
"""

import hashlib
import inspect
import textwrap
from types import MethodType

import torch


def install_frozen_moe(model):
    from liger_kernel.ops import fused_moe

    from arcforge.train.preflight import assert_liger_moe_patch

    # Stock 0.8.1 dX is 55-65% wrong on sm_100 (patches/liger-kernel-0.8.1/0002); a fresh venv
    # without the patch trains silently on that gradient (check_moe_dx_oracle.py, 2026-10-06).
    assert_liger_moe_patch()

    source = textwrap.dedent(inspect.getsource(fused_moe.LigerFusedMoEFunction.backward))
    digest = hashlib.sha256(source.encode()).hexdigest()
    expected = "d301dfbc0a2cd8318290c03503c57033383cf9d0e63eb3aba0752d558fd183f8"
    if digest != expected:
        raise RuntimeError(f"Liger backward changed ({digest}); re-audit before using frozen optimization")
    start = source.index("    # dW2 =")
    end = source.index("    # dx_expanded =", start)
    source = source[:start] + "    ddown_proj = None\n\n" + source[end:]
    start = source.index("    # dW1 =")
    end = source.index("    return dx, dgate_up_proj, ddown_proj, None, dS.view(T, K)", start)
    source = source[:start] + "    dgate_up_proj = None\n\n" + source[end:]
    source = source.replace("@staticmethod\n", "")
    if "_moe_bwd_dW1_kernel[" in source or "_moe_bwd_dW2_kernel[" in source:
        raise RuntimeError("frozen MoE transform failed")
    namespace = dict(vars(fused_moe))
    exec(compile(source, "<arcforge-frozen-moe-backward>", "exec"), namespace)
    backward_impl = namespace["backward"]

    class FrozenMoE(torch.autograd.Function):
        forward = staticmethod(fused_moe.LigerFusedMoEFunction.forward)

        @staticmethod
        def backward(ctx, output_gradient):
            if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                raise RuntimeError("FrozenMoE cannot train expert weights")
            return backward_impl(ctx, output_gradient)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if self.gate_up_proj.requires_grad or self.down_proj.requires_grad:
            raise RuntimeError("expert weights must remain frozen")
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_dim)
        indices = top_k_index.reshape(-1, top_k_index.shape[-1]).to(torch.int32)
        scores = top_k_weights.reshape(-1, top_k_weights.shape[-1])
        return FrozenMoE.apply(x, self.gate_up_proj, self.down_proj, indices, scores).reshape(shape)

    for layer in model.model.layers:
        layer.mlp.experts.forward = MethodType(forward, layer.mlp.experts)
    return {"upstream_backward_sha256": digest, "expert_layers": len(model.model.layers)}, FrozenMoE
