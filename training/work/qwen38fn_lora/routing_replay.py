"""MoE routing replay from the rollout: every MoE layer takes the top-k experts SGLang chose for that token.

The rows come from the TITO ledger (``prepare_tito_mm.py --routing``; arcforge.tito.ledger.assemble_routing):
``routing[p, layer]`` is the experts of the forward that read token p as input, global logical ids. QSA block
selection is not replayed (decision_replay.py can force it for kernel comparisons, not from the ledger).

Under ``force`` the gate weights are the current router softmax at the recorded experts, renormalized like the
model's own top-k (Qwen4ExpTextTopKRouter: softmax over all experts, top-k, divide by their sum), so the router
logits stay differentiable and hidden-state gradients flow through the weights as before. A row of -1 (the last
position of a segment, never forwarded at rollout time) takes the natural top-k. ``observe`` routes naturally and
only counts how often the natural choice equals the recorded one. Both modes count per layer, outside the checkpoint
recompute: rows compared, rows whose top-k set equals the recorded set, and the summed overlap of the two sets.
"""

from types import MethodType

import torch


class RoutingReplay:
    def __init__(self, mode, layers, top_k):
        if mode not in ('observe', 'force'):
            raise ValueError(f'routing replay mode {mode!r}')
        self.mode, self.layers, self.top_k = mode, layers, top_k
        self.routing = None
        self.stats = None

    def install(self, text_model):
        if len(text_model.layers) != self.layers:
            raise RuntimeError(f'{len(text_model.layers)} decoder layers, routing has {self.layers}')
        replay = self
        for index, layer in enumerate(text_model.layers):
            gate = layer.mlp.gate
            if gate.top_k != self.top_k or not gate.norm_topk_prob:
                raise RuntimeError(f'layer {index}: router top_k={gate.top_k} norm_topk_prob={gate.norm_topk_prob}; '
                                   f'replay assumes top_k={self.top_k} with renormalization')

            def forward(self, hidden_states, _index=index):
                hidden = hidden_states.reshape(-1, self.hidden_dim)
                logits = torch.nn.functional.linear(hidden, self.weight)
                probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
                natural = torch.topk(probs, self.top_k, dim=-1).indices
                recorded = replay.rows(_index, hidden.shape[0])
                valid = recorded[:, 0] >= 0
                if torch._C._current_graph_task_id() == -1:  # not the checkpoint recompute of this forward
                    replay.count(_index, natural, recorded, valid)
                indices = torch.where(valid[:, None], recorded, natural) if replay.mode == 'force' else natural
                values = probs.gather(1, indices)
                values = values / values.sum(dim=-1, keepdim=True)
                return logits, values.to(logits.dtype), indices

            gate.forward = MethodType(forward, gate)

    def begin(self, routing):
        """``routing``: (tokens, layers, top_k) int16 on the device, in pack order."""
        if self.routing is not None:
            raise RuntimeError('routing replay: begin() without finish()')
        if routing.dim() != 3 or tuple(routing.shape[1:]) != (self.layers, self.top_k) or routing.dtype != torch.int16:
            raise ValueError(f'routing {routing.dtype}{tuple(routing.shape)}, expected (T, {self.layers}, {self.top_k})')
        self.routing = routing
        zeros = torch.zeros(self.layers, dtype=torch.int64, device=routing.device)
        self.stats = {'rows': zeros.clone(), 'equal': zeros.clone(), 'overlap': zeros.clone(), 'calls': zeros.clone()}

    def rows(self, layer, tokens):
        if self.routing is None:
            raise RuntimeError('routing replay is installed but this forward has no routing (begin() not called)')
        if self.routing.shape[0] != tokens:
            raise RuntimeError(f'layer {layer}: {tokens} tokens, routing has {self.routing.shape[0]} rows')
        return self.routing[:, layer].long()

    def count(self, layer, natural, recorded, valid):
        stats = self.stats
        shared = (natural[:, :, None] == recorded[:, None, :]).any(-1).sum(-1)  # experts in both top-k sets
        stats['rows'][layer] += valid.sum()
        stats['equal'][layer] += (valid & (shared == self.top_k)).sum()
        stats['overlap'][layer] += torch.where(valid, shared, 0).sum()
        stats['calls'][layer] += 1

    def finish(self):
        """Per-layer counts of this forward (one host sync), then forget the routing."""
        if self.routing is None:
            raise RuntimeError('routing replay: finish() without begin()')
        stats = {k: v.cpu() for k, v in self.stats.items()}
        self.routing = self.stats = None
        if (stats['calls'] != 1).any():
            raise RuntimeError(f'MoE gate calls per layer {stats["calls"].tolist()}; expected one forward each')
        return stats


def summarize(stats_list, top_k):
    """Across forwards: per layer and overall, the fraction of rows whose natural top-k equals the recorded set and the
    mean shared experts per row (out of top_k)."""
    rows = sum(s['rows'] for s in stats_list)
    equal = sum(s['equal'] for s in stats_list)
    overlap = sum(s['overlap'] for s in stats_list)
    if int(rows.min()) == 0:
        raise RuntimeError('a layer compared no rows')
    return {'rows_per_layer': int(rows[0]),
            'equal_rate': float(equal.sum() / rows.sum()), 'mean_shared': float(overlap.sum() / rows.sum()),
            'equal_rate_by_layer': (equal / rows).tolist(), 'mean_shared_by_layer': (overlap / rows).tolist(),
            'top_k': top_k}
