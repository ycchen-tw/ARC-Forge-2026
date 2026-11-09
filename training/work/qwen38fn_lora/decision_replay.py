"""Record or force every discrete decision of a forward: MoE top-k experts and QSA block choices.

Whole-model gradients of one long sequence are chaotic under tiny numeric changes: a hard top-k
(MoE routing, QSA block selection) flips somewhere and every layer below sees a different
backward. Comparing two kernels end to end therefore needs both runs to take the same
decisions. ``record`` stores them per rank and per step (keyed by the rank's segment keys);
``replay`` forces them, so the only remaining difference is continuous arithmetic.

Forced MoE routing keeps the router differentiable: the weights are the current softmax at the
recorded experts (renormalized like the original). Forced QSA selections feed both the gather
kernel (compact rows) and the Flex masks (the equivalent boolean table).
"""

import json
from pathlib import Path
from types import MethodType

import torch


class Decisions:
    def __init__(self, mode, directory, rank):
        if mode not in ('record', 'replay'):
            raise ValueError(f'decision mode {mode!r}')
        self.mode, self.rank = mode, rank
        self.directory = Path(directory)
        self.current = None
        self.signature = None
        self.counts = {'moe_forced': 0, 'moe_recorded': 0, 'moe_recompute_checked': 0,
                       'qsa_forced': 0, 'qsa_recorded': 0}

    def path(self, signature):
        return self.directory / f'rank{self.rank}-{signature}.pt'

    def begin(self, keys):
        import hashlib
        self.signature = hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:16]
        path = self.path(self.signature)
        if self.mode == 'replay':
            if not path.exists():
                raise FileNotFoundError(f'no recorded decisions for this step: {path}')
            self.current = torch.load(path, map_location='cpu')
            self.current['source'] = 'loaded'
        else:
            self.current = {'moe': {}, 'qsa': {}, 'source': 'new' if not path.exists() else 'exists'}

    def finish(self):
        if self.mode == 'record' and self.current['source'] == 'new':
            self.directory.mkdir(parents=True, exist_ok=True)
            if len(self.current['moe']) != 48:
                raise RuntimeError(f'recorded MoE decisions for {len(self.current["moe"])} layers, expected 48')
            torch.save({k: v for k, v in self.current.items() if k != 'source'}, self.path(self.signature))
        self.current = None

    def install(self, text_model):
        import sparse_qsa

        decisions = self
        for index, layer in enumerate(text_model.layers):
            gate = layer.mlp.gate

            def forward(self, hidden_states, _index=index):
                hidden = hidden_states.reshape(-1, self.hidden_dim)
                logits = torch.nn.functional.linear(hidden, self.weight)
                probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
                recompute = torch._C._current_graph_task_id() != -1
                state = decisions.current
                if decisions.mode == 'replay':
                    indices = state['moe'][_index].to(device=probs.device, dtype=torch.long)
                    decisions.counts['moe_forced'] += 1
                else:
                    indices = torch.topk(probs, self.top_k, dim=-1).indices
                    if recompute:
                        if not torch.equal(state['moe'][_index].to(indices.device), indices.to(torch.int16)):
                            raise RuntimeError(f'MoE routing of layer {_index} changed between forward and recompute')
                        decisions.counts['moe_recompute_checked'] += 1
                    else:
                        if _index in state['moe']:
                            raise RuntimeError(f'MoE layer {_index} recorded twice in one step')
                        state['moe'][_index] = indices.to(torch.int16).cpu()
                        decisions.counts['moe_recorded'] += 1
                if indices.shape != (hidden.shape[0], self.top_k):
                    raise RuntimeError(f'routing shape {tuple(indices.shape)} for {hidden.shape[0]} tokens')
                values = probs.gather(1, indices)
                if self.norm_topk_prob:
                    values = values / values.sum(dim=-1, keepdim=True)
                return logits, values.to(logits.dtype), indices

            gate.forward = MethodType(forward, gate)

        original_compact = sparse_qsa.select_compact
        original_blocks = sparse_qsa.select_blocks

        def recorded(indexer, meta):
            from qsa_gather import QSASelection
            blocks, counts = decisions.current['qsa'][indexer.layer_idx]
            decisions.counts['qsa_forced'] += 1
            if 'qsa_audit' in meta:  # the trainer's every-layer-ran check
                meta['qsa_audit'][indexer.layer_idx] = {'forced_from': decisions.signature}
            device = indexer.index_qk_proj.weight.device
            return QSASelection(blocks.to(device), counts.to(device))

        def select_compact(indexer, hidden, position_embeddings, meta, chunk_size=256):
            if decisions.mode == 'replay':
                return recorded(indexer, meta)
            selection = original_compact(indexer, hidden, position_embeddings, meta, chunk_size)
            decisions.current['qsa'][indexer.layer_idx] = (selection.blocks.cpu(), selection.counts.cpu())
            decisions.counts['qsa_recorded'] += 1
            return selection

        def select_blocks(indexer, hidden, position_embeddings, meta, chunk_size=256):
            if decisions.mode == 'record':
                raise RuntimeError('record decisions with the gather backend (compact selections)')
            selection = recorded(indexer, meta)
            # Boolean table with the same visible keys: selected complete blocks plus the tail block,
            # which the Flex masks OR in themselves (causality is enforced by the mask either way).
            table = torch.zeros((meta['total'], meta['max_length'] // 4 + 1), device=selection.blocks.device,
                                dtype=torch.bool)
            width = selection.blocks.shape[1]
            valid = torch.arange(width, device=table.device)[None, :] < selection.counts[:, None]
            rows = torch.arange(meta['total'], device=table.device)[:, None].expand(-1, width)
            table[rows[valid], selection.blocks.long()[valid]] = True
            return table

        sparse_qsa.select_compact = select_compact
        sparse_qsa.select_blocks = select_blocks
        self.originals = (original_compact, original_blocks)
