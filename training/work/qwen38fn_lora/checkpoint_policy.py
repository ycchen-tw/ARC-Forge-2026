"""Reversible whole-layer versus MoE-only activation recomputation."""

from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint


def install_checkpoint_policy(model):
    layers = model.base_model.model.model.layers
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    policy = {'mode': 'layer', 'mlp_checkpoint_calls': 0}
    for index, layer in enumerate(layers):
        original = layer.mlp.forward

        def forward(self, hidden, original=original, index=index):
            selected = policy['mode'] == 'moe' or (policy['mode'] == 'hybrid' and index % 4 == 0)
            if not selected or not self.training or not torch.is_grad_enabled():
                return original(hidden)
            policy['mlp_checkpoint_calls'] += 1
            return checkpoint(original, hidden, use_reentrant=False)

        layer.mlp.forward = MethodType(forward, layer.mlp)

    def set_policy(mode):
        if mode not in ('layer', 'moe', 'hybrid'):
            raise ValueError(mode)
        policy['mode'] = mode
        for index, layer in enumerate(layers):
            layer.gradient_checkpointing = mode == 'layer' or (mode == 'hybrid' and index % 4 != 0)

    return set_policy, policy


def set_prefetch(model, depth):
    if depth not in (1,2,4):
        raise ValueError(depth)
    layers = model.base_model.model.model.layers
    for index, layer in enumerate(layers):
        layer.set_modules_to_forward_prefetch(list(layers[index+1:index+1+depth]))
        layer.set_modules_to_backward_prefetch(list(reversed(layers[max(0,index-depth):index])))
