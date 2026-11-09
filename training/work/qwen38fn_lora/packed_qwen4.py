"""Qwen4 text packing with FA4/GDN segment boundaries and isolated PLE.

Short QSA segments use causal FA4; longer segments retain QSA top-k via Flex.
PLE alone is reshaped back to padded examples for its n-gram hashing and dilated
convolution. This preserves those boundaries without writing another kernel.
"""

from contextlib import nullcontext
from contextvars import ContextVar
from types import MethodType

import numpy as np
import torch
from qsa_gather import SelectionReplay
from viggo_data import render_training

from arcforge.train.packed_text import PackedTextCollator

_PACK = ContextVar('qwen4_packed_metadata', default=None)


class RuntimeRecords:
    """An opaque pytree leaf: FSDP must not copy this per-forward collector."""
    def __init__(self):
        self.records={}

    def __len__(self):
        return len(self.records)

    def __contains__(self,key):
        return key in self.records

    def __getitem__(self,key):
        return self.records[key]

    def __setitem__(self,key,value):
        self.records[key]=value

    def values(self):
        return self.records.values()


def enable_fp32_router(model):
    """Keep router logits unrounded before discrete top-k; expert scores stay BF16."""
    def forward(self, hidden_states):
        hidden = hidden_states.reshape(-1, self.hidden_dim)
        logits = torch.nn.functional.linear(hidden.float(), self.weight.float())
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        values, indices = torch.topk(probabilities, self.top_k, dim=-1)
        if self.norm_topk_prob:
            values = values / values.sum(dim=-1, keepdim=True)
        return logits, values.to(hidden_states.dtype), indices
    for layer in model.model.layers:
        layer.mlp.gate.forward = MethodType(forward, layer.mlp.gate)


def pack_rows(tokenizer, rows, sequence_cap, device):
    segments = [render_training(tokenizer, row, sequence_cap) for row in rows]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return pack_segments(segments, pad, device)


def pack_segments(segments, pad_token_id, device):
    arrays = [(np.asarray(ids, dtype=np.int64), np.asarray(labels, dtype=np.int64)) for ids, labels in segments]
    batch = PackedTextCollator(pad_token_id=pad_token_id, device=device)(arrays)
    # Per-example causal LM never supervises its first token.
    batch['labels'][0, batch['cu_seq_lens_q'][:-1].long()] = -100
    row_indices = batch['seq_idx'][0].long()
    positions = batch['position_ids'][0, 0]
    lengths = [len(ids) for ids, _ in segments]
    batch['packed_meta'] = {'rows': row_indices, 'positions': positions, 'segments': len(segments),
                            'max_length': max(lengths), 'total': sum(lengths), 'lengths':lengths,
                            'train_indexer':False,
                            'attention_mode':'qsa',
                            'qsa_mask_backend':'generic',
                            'ple_mode':'packed',  # ple_packed.py; 'padded' = the old per-row batch
                            'ple_crosscheck':None,  # debug: a RuntimeRecords to compare packed vs padded PLE
                            'qsa_replay':SelectionReplay(),
                            'cu': batch['cu_seq_lens_q'],
                            'qsa_mask':torch.ones((1,1,1,1),device=device,dtype=torch.bool)}
    # Bypass the model's dense mask construction. Our packed attention dispatcher
    # requires metadata and uses cu_seqlens, never this placeholder mask.
    # Vanilla Qwen4 will fail on this None mask rather than silently treating the
    # whole pack as one sample. The installed attention wrapper replaces it.
    batch['attention_mask'] = {'full_attention': None, 'linear_attention': None}
    return batch


def install_packing(model):
    from flash_attn.cute.interface import flash_attn_varlen_func
    from transformers.models.qwen4_exp.modeling_qwen4_exp import ALL_ATTENTION_FUNCTIONS, Qwen4ExpTextAttention

    report = {'fa4_calls': 0, 'ple_calls': 0, 'indexer_bypasses': 0, 'sparse_flex_calls':0, 'sparse_block_calls':0,
              'sparse_gather_calls':0,
              'full_flex_calls':0, 'full_attention_bypasses':0,
              'sparse_indexer_calls':0, 'indexer_kl_calls':0,
              'unpacked_fa4': False, 'unpacked_fa4_calls': 0}
    original_sdpa = ALL_ATTENTION_FUNCTIONS['sdpa']

    def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        if not isinstance(module,Qwen4ExpTextAttention):
            return original_sdpa(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)
        meta = kwargs.get('packed_meta')
        if meta is None:
            if report['unpacked_fa4'] and query.shape[-2] == key.shape[-2]:
                visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
                lengths = visible[:,0,-1,:].sum(-1).tolist()
                width = query.shape[-2]
                if max(lengths) > module.indexer.block_topk * module.indexer.compress_ratio:
                    raise RuntimeError('FA4 reference requires all-visible short QSA')
                indices = torch.cat([torch.arange(n,device=query.device)+i*width for i,n in enumerate(lengths)])
                cu = torch.tensor([0]+list(np.cumsum(lengths)),device=query.device,dtype=torch.int32)
                def compact(x):
                    return x.transpose(1,2).reshape(-1,x.shape[1],x.shape[-1])[indices].contiguous()
                result,_ = flash_attn_varlen_func(compact(query),compact(key),compact(value),
                    cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=max(lengths),max_seqlen_k=max(lengths),
                    causal=True,softmax_scale=scaling)
                output = result.new_zeros((query.shape[0]*width,query.shape[1],query.shape[-1]))
                output[indices] = result
                report['unpacked_fa4_calls'] += 1
                return output.view(query.shape[0],width,query.shape[1],query.shape[-1]),None
            if attention_mask is not None and attention_mask.shape[-1] == 1 and query.shape[-2] != 1:
                raise RuntimeError('packing metadata lost before attention')
            return original_sdpa(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        if query.shape[0] != 1 or query.shape[-2] != meta['total'] or dropout != 0:
            raise RuntimeError('packed attention requires B=1, complete metadata and dropout=0')
        auxiliary=None
        if meta['train_indexer']:
            from qsa_indexer_loss import indexer_kl
            auxiliary=indexer_kl(module.indexer,meta['qsa_hidden'],meta['qsa_position_embeddings'],
                                query,key,meta['qsa_selection'],meta,scaling)
            # Recompute the same graph during GC, but do not retain replay graphs.
            if module.layer_idx not in meta['qsa_aux_losses']:
                meta['qsa_aux_losses'][module.layer_idx]=auxiliary.detach()
            report['indexer_kl_calls']+=1
        mode = meta['attention_mode']
        import qsa_gather
        if qsa_gather.KV_FP8['on'] and mode != 'qsa':
            raise ValueError(f'--qsa-kv-fp8 covers the QSA attention mode only, not {mode!r}')
        if mode == 'full_flex':
            from sparse_qsa import full_attention
            output,_=full_attention(query,key,value,meta,scaling)
            report['full_flex_calls'] += 1
        elif mode == 'qsa' and meta['max_length'] > module.indexer.block_topk * module.indexer.compress_ratio:
            from sparse_qsa import sparse_attention
            report[{'block_sparse':'sparse_block_calls','gather':'sparse_gather_calls'}.get(
                meta['qsa_mask_backend'],'sparse_flex_calls')] += 1
            output,_=sparse_attention(query,key,value,meta['qsa_selection'],meta,module.indexer.compress_ratio,scaling,
                                      mask_backend=meta['qsa_mask_backend'])
        else:
            key, value = qsa_gather.rollout_kv(key, value)   # every segment within the budget: dense, FP8 K/V if set
            output, _ = flash_attn_varlen_func(
                query[0].transpose(0, 1).contiguous(), key[0].transpose(0, 1).contiguous(),
                value[0].transpose(0, 1).contiguous(), cu_seqlens_q=meta['cu'], cu_seqlens_k=meta['cu'],
                max_seqlen_q=meta['max_length'], max_seqlen_k=meta['max_length'],
                causal=True, softmax_scale=scaling)
            report['fa4_calls'] += 1
            output=output.unsqueeze(0)
        if auxiliary is not None and torch.is_grad_enabled():
            from qsa_indexer_loss import AttachIndexerLoss
            output=AttachIndexerLoss.apply(output,auxiliary,meta['qsa_aux_grad_scale'])
        return output,None

    ALL_ATTENTION_FUNCTIONS.register('sdpa', attention)
    model._arc_packing_report = report
    for layer in model.model.layers:
        # what the wrapper calls; trainer_kernels' hc_inject replaces it and this wrapper stays outermost
        layer.packed_inner_forward = layer.forward

        def layer_forward(self, *args, **kwargs):
            # This wrapper also runs during non-reentrant checkpoint recompute,
            # so PLE/indexer never rely on stale state from the outer forward.
            token = _PACK.set(kwargs.get('packed_meta'))
            try:
                return self.packed_inner_forward(*args, **kwargs)
            finally:
                _PACK.reset(token)

        layer.forward = MethodType(layer_forward, layer)
        if layer.layer_type == 'qwen_sparse_attention':
            original_attention = layer.self_attn.forward

            def attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                                  original=original_attention, **kwargs):
                meta = kwargs.get('packed_meta')
                if meta is not None:
                    if attention_mask is not None:
                        raise RuntimeError('packed input must use the guarded mask dictionary')
                    attention_mask = meta['qsa_mask']
                    if past_key_values is not None:
                        raise RuntimeError('packed QSA training requires no cache')
                    # Per-call state, never mutate shared batch metadata across layers or GC.
                    meta = dict(meta)
                    if meta['attention_mode'] not in ('qsa','full_flex','full_fa4'):
                        raise ValueError('unknown attention ablation mode')
                    if meta['attention_mode'] != 'qsa' and meta['train_indexer']:
                        raise ValueError('full attention ablation requires frozen indexer / no KL')
                    if meta['train_indexer']:
                        meta['qsa_hidden']=hidden_states
                        meta['qsa_position_embeddings']=position_embeddings
                    if meta['attention_mode'] == 'full_flex' or (meta['attention_mode'] == 'qsa' and meta['qsa_mask_backend'] != 'gather'
                                                                 and meta['max_length'] > self.indexer.block_topk*self.indexer.compress_ratio):
                        from sparse_qsa import block_layout
                        meta['qsa_block_layout'] = block_layout(meta['lengths'],hidden_states.device)
                    kwargs['packed_meta'] = meta
                token = _PACK.set(meta)
                try:
                    return original(hidden_states,position_embeddings,attention_mask,past_key_values=past_key_values,**kwargs)
                finally:
                    _PACK.reset(token)

            layer.self_attn.forward = MethodType(attention_forward,layer.self_attn)
            indexer = layer.self_attn.indexer
            original_indexer = indexer.forward

            def indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, original=original_indexer):
                meta = _PACK.get()
                if meta is None:
                    return original(hidden_states, position_embeddings, attention_mask, past_key_values)
                if past_key_values is not None:
                    raise RuntimeError('packed QSA only supports no-cache training')
                if meta['attention_mode'] != 'qsa':
                    if meta['train_indexer'] or any(p.requires_grad for p in self.parameters()):
                        raise RuntimeError('full-attention ablation cannot train the indexer')
                    report['full_attention_bypasses'] += 1
                    return attention_mask
                if meta['qsa_mask_backend'] == 'gather' and meta['max_length'] > self.block_topk * self.compress_ratio:
                    if meta['train_indexer']:
                        raise ValueError('gather QSA backend requires a frozen indexer (no KL objective)')
                    from sparse_qsa import select_compact

                    def compute(self=self):
                        report['sparse_indexer_calls'] += 1
                        return select_compact(self,hidden_states,position_embeddings,meta)
                    # Forward records, checkpoint recompute replays: one indexer pass per step.
                    meta['qsa_selection'] = meta['qsa_replay'].select(self.layer_idx,compute)
                    return attention_mask
                if meta['max_length'] > self.block_topk * self.compress_ratio or meta['train_indexer']:
                    from sparse_qsa import select_blocks
                    meta['qsa_selection'] = select_blocks(self,hidden_states,position_embeddings,meta)
                    report['sparse_indexer_calls'] += 1
                    return attention_mask
                report['indexer_bypasses'] += 1
                return attention_mask

            indexer.forward = MethodType(indexer_forward, indexer)
        if layer.ple is not None:
            original_ple = layer.ple.forward

            def ple_forward(self, hidden_states, input_ids, past_key_values, conv_mask=None, original=original_ple):
                meta = _PACK.get()
                if meta is None:
                    return original(hidden_states, input_ids, past_key_values, conv_mask=conv_mask)
                if past_key_values is not None or hidden_states.shape[0] != 1:
                    raise RuntimeError('packed PLE requires no cache and B=1')
                if meta['ple_mode'] not in ('packed', 'padded'):
                    raise ValueError(f'unknown PLE mode {meta["ple_mode"]!r}')
                packed = None
                if meta['ple_mode'] == 'packed':
                    from ple_packed import packed_ple_forward
                    report['ple_calls'] += 1
                    packed = packed_ple_forward(self, hidden_states, input_ids, meta['lengths'])
                    if meta['ple_crosscheck'] is None or torch._C._current_graph_task_id() != -1:
                        return packed
                rows, cols = meta['rows'], meta['positions']
                shape = (meta['segments'], meta['max_length'])
                # A crosscheck must not save tensors: the checkpoint recompute runs the packed path only.
                with torch.no_grad() if packed is not None else nullcontext():
                    unpacked_hidden = hidden_states.new_zeros((*shape, hidden_states.shape[-1]))
                    unpacked_hidden[rows, cols] = hidden_states[0]
                    eos = self.ple_embedding.eos_token_id
                    unpacked_ids = input_ids.new_full(shape, eos)
                    unpacked_ids[rows, cols] = input_ids[0]
                    mask = input_ids.new_zeros(shape)
                    mask[rows, cols] = 1
                    unpacked = original(unpacked_hidden, unpacked_ids, None, conv_mask=mask)
                    padded = unpacked[rows, cols].unsqueeze(0)
                if packed is None:
                    report['ple_calls'] += 1
                    return padded
                with torch.no_grad():
                    difference = (packed.detach().float() - padded.float()).abs()
                meta['ple_crosscheck'][len(meta['ple_crosscheck'])] = {
                    'max_abs': float(difference.max()), 'mean_abs': float(difference.mean()),
                    'reference_mean_abs': float(padded.float().abs().mean()), 'segments': len(meta['lengths'])}
                return packed

            layer.ple.forward = MethodType(ple_forward, layer.ple)
    return report
