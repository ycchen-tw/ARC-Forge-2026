"""Native Qwen4 image graph with the audited text kernels and replicated LoRA.

No image embedding cache: each image forward executes the original visual tower.
Sequence-boundary positions for PLE remain separate from multimodal RoPE.
"""

import json
from collections import defaultdict
from pathlib import Path
from types import MethodType

import torch
from packed_qwen4 import RuntimeRecords, pack_segments


def load_multimodal(model_path,base_storage='bf16',device=None,ple_codec='fp8'):
    from fp8_base import install_expert_backend
    from packed_qwen4 import install_packing
    from probe_train import apply_qwen4_liger
    from qsa_fastpath import install_short_qsa
    from train_viggo_fsdp import load_cpu_model
    from transformers import AutoConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration

    text, seconds, scales, ple_scale = load_cpu_model(model_path,base_storage,device,ple_codec)
    apply_qwen4_liger(text)
    install_short_qsa(text)
    kernels = install_packing(text)
    install_expert_backend(text,base_storage)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config.text_config = text.config
    visual, state = load_vision_tower(model_path, config)
    # Allocate only the native module skeleton, then replace all parameter owners.
    with torch.device('meta'):
        model = Qwen4ExpForConditionalGeneration(config)
    model.model.language_model = text.model
    model.model.visual = visual
    model.lm_head = text.lm_head
    if any(value.is_meta for value in list(model.parameters())+list(model.buffers())):
        raise RuntimeError('unmaterialized multimodal checkpoint tensor')
    model.forward = MethodType(training_forward, model)
    return model, {'load_seconds':seconds, 'scales':scales, 'ple_scale':ple_scale,
                   'visual_tensors_loaded_bitwise':len(state),'base_storage':base_storage}, kernels


def load_vision_tower(model_path, config=None):
    """The checkpoint's ViT (``model.visual.*``, BF16, SDPA attention) on the CPU, loaded bit for bit; and its state."""
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    if config is None:
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config.vision_config._attn_implementation = 'sdpa'
    visual = Qwen4ExpVisionModel(config.vision_config).to(dtype=torch.bfloat16)
    index = json.loads((Path(model_path)/'model.safetensors.index.json').read_text())['weight_map']
    shards = defaultdict(list)
    for key, filename in index.items():
        if key.startswith('model.visual.'):
            shards[filename].append(key)
    state = {}
    for filename, keys in shards.items():
        with safe_open(Path(model_path)/filename, framework='pt', device='cpu') as handle:
            for key in keys:
                state[key.removeprefix('model.visual.')] = handle.get_tensor(key)
    visual.load_state_dict(state, strict=True)
    if any(not torch.equal(value, visual.state_dict()[key]) for key, value in state.items()):
        raise RuntimeError('ViT checkpoint load was not bitwise exact')
    return visual, state


def training_forward(self, input_ids=None, labels=None, measure_accuracy=False, **kwargs):
    """Native multimodal backbone; exact masked CE on supervised positions only."""
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    from overfit_metrics import score_hidden
    from transformers.modeling_outputs import CausalLMOutputWithPast

    if labels is None or input_ids is None or kwargs.get('use_cache', False):
        raise ValueError('training forward requires IDs, labels and no cache')
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden = outputs.last_hidden_state
    target = labels[:, 1:].reshape(-1)
    valid = target != -100
    shifted = hidden[:, :-1].reshape(-1, hidden.shape[-1])[valid].contiguous()
    loss = LigerFusedLinearCrossEntropyLoss()(self.lm_head.weight, shifted, target[valid])
    result = CausalLMOutputWithPast(loss=loss)
    if measure_accuracy:
        result['metrics'] = score_hidden(hidden, self.lm_head.weight, labels)
    return result


def make_batch(row, pixels, native_model, pad_token, device):
    batch = pack_segments([(row['input_ids'], row['labels'])], pad_token, device)
    meta = batch['packed_meta']
    meta.update(train_indexer=True, qsa_audit=RuntimeRecords(), qsa_aux_losses=RuntimeRecords())
    ids = batch['input_ids']
    mm_types = torch.tensor([row['mm_token_type_ids']], device=device)
    grid = torch.tensor(row['image_grid_thw'], dtype=torch.long, device=device) if row['images'] else None
    # Do not pass the packing mask dict to get_rope_index: it requires a 2D mask.
    position_ids, _ = native_model.model.get_rope_index(ids, mm_types, image_grid_thw=grid)
    batch['position_ids'] = position_ids
    batch['mm_token_type_ids'] = mm_types
    if row['images']:
        batch['pixel_values'] = pixels.to(device=device, dtype=torch.bfloat16)
        batch['image_grid_thw'] = grid
    return batch


def shard_multimodal(model, parameters, mesh, device):
    from distributed_utils import RowShardedEmbedding
    from fp8_base import RowShardedPLE
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    base = model.base_model.model
    embedding = base.model.language_model.layers[1].ple.ple_embedding
    if not isinstance(embedding.ngram_embedding,RowShardedPLE):
        embedding.ngram_embedding = RowShardedEmbedding(embedding.ngram_embedding.weight, device)
    ignored = set(parameters)
    common = dict(mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                  reduce_dtype=torch.bfloat16), reshard_after_forward=True)
    for layer in base.model.language_model.layers:
        if layer.mlp.experts.gate_up_proj.dtype==torch.float8_e4m3fn:
            fully_shard(layer.mlp.experts,mesh=mesh,mp_policy=MixedPrecisionPolicy(),reshard_after_forward=True)
        fully_shard(layer, ignored_params=set(layer.parameters()) & ignored, **common)
    # Replicate the ~0.5B vision tower; mixed text/image ranks must never perform
    # mismatched conditional FSDP collectives. Its LoRA uses explicit reductions.
    visual_params = set(base.model.visual.parameters())
    base.model.visual.to(device=device)
    fully_shard(model, ignored_params=ignored | visual_params, **common)
    for buffer in model.buffers():
        if buffer.device.type != 'cuda':
            buffer.data = buffer.data.to(device)
    if any(type(p) is not torch.nn.Parameter for p in parameters):
        raise RuntimeError('LoRA unexpectedly sharded')


def adapter_checkpoint_key(name):
    prefix = 'base_model.model.model.'
    if not name.startswith(prefix) or not name.endswith('.lora_A.weight'):
        raise ValueError(f'unexpected adapter key: {name}')
    tail = name[len(prefix):].removesuffix('.lora_A.weight')+'.weight'
    if tail.startswith(('language_model.', 'visual.')):
        return 'model.'+tail
    if tail.startswith('layers.'):
        return 'model.language_model.'+tail
    raise ValueError(f'unhandled adapter module: {name}')
