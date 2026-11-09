"""Exact short-context QSA shortcut: selecting all blocks needs no indexer.

For KV length <= floor(budget/ratio)*ratio, every complete block and the
incomplete tail are selected, independently of scores. Only use without cache:
cached generation must still update the indexer for later, longer contexts.
"""

from types import MethodType

import torch


def install_short_qsa(model):
    report = {"layers": 0, "fast_calls": 0, "fallback_calls": 0, "max_fast_kv": 0}
    for layer in model.model.layers:
        if layer.layer_type != "qwen_sparse_attention":
            continue
        indexer = layer.self_attn.indexer
        original = indexer.forward

        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, original=original):
            complete_budget = self.block_topk * self.compress_ratio
            if past_key_values is None and attention_mask.shape[-1] <= complete_budget:
                report["fast_calls"] += 1
                report["max_fast_kv"] = max(report["max_fast_kv"], attention_mask.shape[-1])
                if attention_mask.dtype == torch.bool:
                    return attention_mask
                return torch.where(attention_mask == 0, attention_mask.new_zeros(()),
                                   torch.finfo(attention_mask.dtype).min)
            report["fallback_calls"] += 1
            return original(hidden_states, position_embeddings, attention_mask, past_key_values)

        indexer.forward = MethodType(forward, indexer)
        report["layers"] += 1
    if report["layers"] != 12:
        raise RuntimeError(f"expected 12 QSA layers, got {report['layers']}")
    return report
