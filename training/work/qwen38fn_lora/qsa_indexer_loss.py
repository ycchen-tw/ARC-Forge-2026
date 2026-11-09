"""Stage-2 QSA indexer KL: selected-block, max-pooled attention teacher.

The backbone teacher and indexer input are detached. Only indexer parameters
receive this auxiliary gradient. No query subsampling: every query with a
complete visible block contributes, normalized by the pack's token count.
"""

import math

import torch


@torch.no_grad()
def teacher_targets(query,key,chosen,valid,positions,ratio,scale):
    """Per-query gather keeps temporary storage proportional to selected keys."""
    count,blocks=chosen.shape
    heads,groups=key.shape[1],query.shape[1]//key.shape[1]
    indices=(chosen[:,:,None]*ratio+torch.arange(ratio,device=query.device)).flatten(1)
    token_valid=valid[:,:,None].expand(-1,-1,ratio).flatten(1)
    tails=((positions+1)//ratio)[:,None]*ratio+torch.arange(ratio-1,device=query.device)
    tail_valid=tails<=positions[:,None]
    indices=torch.cat((indices,tails),dim=-1)
    token_valid=torch.cat((token_valid,tail_valid),dim=-1)
    gathered=key[indices.clamp(max=len(key)-1)].float().permute(0,2,3,1)
    scores=query.float().view(count,heads,groups,-1)@gathered
    scores=(scores*scale).masked_fill(~token_valid[:,None,None,:],float('-inf'))
    probabilities=scores.softmax(-1).mean((1,2))
    pooled=probabilities[:,:blocks*ratio].view(count,blocks,ratio).amax(-1)*valid
    return pooled/pooled.sum(-1,keepdim=True).clamp_min(1e-30)


def indexer_kl(indexer,hidden,position_embeddings,query,key,table,meta,scale,chunk_size=64):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb
    ratio=indexer.compress_ratio
    hidden=hidden.detach()
    projected=indexer.index_qk_proj(hidden)
    q,raw=projected.split([indexer.index_n_heads*indexer.index_head_dim,indexer.index_head_dim],dim=-1)
    q=indexer.q_layernorm(q.view(1,meta['total'],indexer.index_n_heads,indexer.index_head_dim))
    cos,sin=position_embeddings
    q=apply_rotary_pos_emb(q,cos=cos,sin=sin,unsqueeze_dim=2)[0]
    total=projected.sum()*0
    start=0
    for length in meta['lengths']:
        blocks=length//ratio
        if blocks==0:
            start+=length
            continue
        pooled=raw[0,start:start+blocks*ratio].view(blocks,ratio,-1).float().mean(1).to(raw.dtype)
        pooled=indexer.k_layernorm(pooled)
        pooled=apply_rotary_pos_emb(pooled.unsqueeze(1),cos=cos[0,start:start+blocks*ratio:ratio],
                                    sin=sin[0,start:start+blocks*ratio:ratio]).squeeze(1)
        for lo in range(0,length,chunk_size):
            hi=min(lo+chunk_size,length)
            positions=torch.arange(lo,hi,device=hidden.device)
            selected=table[start+lo:start+hi,:blocks]
            chosen=selected.int().topk(min(indexer.block_topk,blocks),dim=-1).indices
            valid=selected.gather(1,chosen)&(chosen<((positions+1)//ratio)[:,None])
            target=teacher_targets(query[0,:,start+lo:start+hi].transpose(0,1),
                                   key[0,:,start:start+length].transpose(0,1),chosen,valid,positions,ratio,scale)
            logits=(q[start+lo:start+hi].float()@pooled[chosen].float().transpose(1,2)).relu().sum(1)
            logits=logits/math.sqrt(indexer.index_head_dim)
            log_probs=logits.masked_fill(~valid,torch.finfo(logits.dtype).min).log_softmax(-1)
            total=total+(target*(target.clamp_min(1e-30).log()-log_probs)).sum()
        start+=length
    return total/meta['total']


class AttachIndexerLoss(torch.autograd.Function):
    """Identity forward, exact explicitly weighted auxiliary backward.

Keeps KL backward inside the owning FSDP layer's normal backward schedule,
instead of unsharding all 12 layers at once from an external scalar loss sum.
The caller MUST supply the complete auxiliary scaling (DP query-token weight,
KL coefficient, layer normalization). BF16 training here uses no GradScaler.
"""

    @staticmethod
    def forward(ctx,output,auxiliary,gradient_scale):
        ctx.save_for_backward(auxiliary)
        ctx.gradient_scale=gradient_scale
        return output

    @staticmethod
    def backward(ctx,gradient):
        auxiliary,=ctx.saved_tensors
        return gradient,torch.ones_like(auxiliary)*ctx.gradient_scale,None
