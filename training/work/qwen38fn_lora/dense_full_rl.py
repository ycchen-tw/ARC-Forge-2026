"""Full-weight training of the RL trainer's projections (train_rl_service.py ``--trainable full``), in place of the LoRA.

The trainable set is the LoRA's (train_tito_mm.lora_targets with the shared expert: the 228 QSA/GDN projections and the
144 shared-expert gate/up/down projections, 372 weights, ~2.92B elements), now the weights themselves. Everything else
stays frozen as in the LoRA trainer (routed INT3 experts, routers, shared-expert gate, PLE, embeddings, lm_head, hyper
connections, norms, indexer, ViT).

- Masters: FP32, sharded by FSDP2 (MixedPrecisionPolicy param_dtype BF16 = what the forward multiplies and what deploys;
  reduce_dtype FP32; gradient divide factor 1 = summed over ranks, as the LoRA trainer's all-reduce). AdamW state FP32,
  sharded with them. A BF16 master would freeze most updates (the step lr ~1e-6 is below the BF16 ulp of |w|).
- The 144 projections the package deploys as block FP8 run work/qwen38fn_qad/dense_full_qat.py: the forward quantizes
  the BF16 cast of the master exactly as the package does (checked bit for bit against --dense-qat-reference at
  install), backward straight-through; the publish writes those same e4m3 / scale_inv bytes (merged_export
  MergedExporter, full mode). The other 228 weights are plain BF16 linears, as deployed.
- ``--freeze-first-layers F``: the trainable projections of decoder layers 0..F-1 stay frozen at the package values.
  The embedding, the PLE table and the ViT are frozen anyway, so no gradient reaches the first F layers at all:
  autograd builds no graph for them, they keep no checkpointed inputs and run no recompute and no backward. F = 2
  also drops layer 1, the PLE layer (config ple_layer_ids [2]), whose backward over the 10,240-wide stream FlashNext-RL
  (arXiv 2610.28998 §3.2.12) found to be the largest transient. Their modules run one forward per micro-step;
  train_tito_mm.assert_step_kernels counts that through ``FullWeights.frozen``.
- Checkpoints: torch.distributed.checkpoint of the trainable masters and AdamW state (~12 bytes per element, ~35 GB),
  every rank writing its shards.
- train_tito_mm ``--trainable full --vit-lora-rank R``: the ViT stays outside FSDP (replicated, as always) and carries a
  vision LoRA whose FP32 A/B are plain replicated tensors: ``shard(vision_lora=...)`` accepts exactly those as the ViT's
  trainable parameters, the caller all-reduces their gradients (sync_gradients, FSDP does not see them) and passes
  them to ``clip_gradients(replicated=...)`` so the norm and the clip cover both sets.
"""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

QAD_DIR = Path(__file__).resolve().parents[1] / 'qwen38fn_qad'
# dense_full_qat.py and what it imports (QAD v2's full-weight block-FP8 QAT); train_rl_service copies them into <out>/code
QAD_MODULES = ('dense_full_qat.py', 'scale_grad_triton.py', 'qad_timing.py')
if str(QAD_DIR) not in sys.path:
    sys.path.append(str(QAD_DIR))

LAYER = re.compile(r'^model\.language_model\.layers\.(\d+)\.')
GRAD_KERNELS = ('quack', 'fused', 'fp32')


def layer_of(name):
    match = LAYER.match(name)
    if match is None:
        raise ValueError(f'{name}: not a decoder-layer parameter')
    return int(match.group(1))


def install(model, targets, package, device, freeze_layers, grad_kernel, grad_precision):
    """Make the ``targets`` (module names, train_tito_mm.lora_targets) of the plain (non-PEFT) ``model`` the trainable
    set; call before FSDP, on the whole BF16 model.

    The package's block-FP8 projections are bound to dense_full_qat (after it checks that quantizing their BF16 weights
    gives the package's bytes); every target outside the first ``freeze_layers`` layers becomes an FP32 Parameter with
    requires_grad, the rest of the model is frozen. Returns FullWeights: ``names`` / ``parameters`` (trainable, model
    order), ``keys`` (the 372 checkpoint weight keys, the merged export's key set), ``dense_fp8_keys``, ``frozen``
    (assert_step_kernels' counts) and ``verified``."""
    import dense_full_qat

    text_config = model.config.text_config
    if not 0 <= freeze_layers < text_config.num_hidden_layers:
        raise ValueError(f'--freeze-first-layers {freeze_layers} of {text_config.num_hidden_layers} layers')
    model.requires_grad_(False)
    modules = dict(model.named_modules())
    for name in targets:
        module = modules.get(name)
        if type(module) is not torch.nn.Linear or module.bias is not None or module.weight.dtype != torch.bfloat16:
            raise RuntimeError(f'{name}: expected a bias-free BF16 nn.Linear, found {type(module).__name__}')
    fp8_modules, verified = dense_full_qat.install(model, package, device)
    if not set(fp8_modules) <= set(targets):
        raise RuntimeError(f'block-FP8 projections outside the trainable set: {sorted(set(fp8_modules) - set(targets))[:3]}')
    dense_full_qat.DENSE_GRAD.update(mode=grad_kernel, precision=grad_precision)
    keys = [name + '.weight' for name in targets]
    trainable = {k for k in keys if layer_of(k) >= freeze_layers}
    names, parameters = [], []
    for name, p in model.named_parameters():
        if name in trainable:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
            names.append(name)
            parameters.append(p)
    if set(names) != trainable:
        raise RuntimeError(f'trainable weights not found as parameters: {sorted(trainable - set(names))[:3]}')
    gdn = sum(1 for i in range(freeze_layers) if text_config.layer_types[i] == 'linear_attention')
    # stream_grad False: the stream entering the first trained layer has no gradient (frozen embeddings; load_full_trainer
    # removes the input-require-grads hook), with or without frozen layers
    frozen = {'layers': freeze_layers, 'gdn': gdn,
              'dense_fp8': sum(1 for m in fp8_modules if layer_of(m) < freeze_layers), 'stream_grad': False}
    return SimpleNamespace(names=names, parameters=parameters, keys=sorted(keys),
                           dense_fp8_keys=sorted(m + '.weight' for m in fp8_modules), frozen=frozen, verified=verified)


def shard(model, mesh, device, vision_lora=()):
    """FSDP2 over the plain model: train_tito_mm's layout (PLE / embedding row shards, FP8 experts their own unit, one
    unit per decoder layer, the ViT replicated outside FSDP), but trainable FP32 masters computing in BF16 with an FP32
    gradient reduce that sums over ranks (no division by the world size: the RL loss is already normalized).
    ``vision_lora``: the ViT's only trainable parameters (train_tito_mm's vision LoRA), replicated with the ViT."""
    from distributed_utils import RowShardedEmbedding
    from fp8_base import RowShardedPLE
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    embedding = model.model.language_model.layers[1].ple.ple_embedding
    if not isinstance(embedding.ngram_embedding, RowShardedPLE):
        embedding.ngram_embedding = RowShardedEmbedding(embedding.ngram_embedding.weight, device)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.language_model.layers:
        if layer.mlp.experts.gate_up_proj.dtype == torch.float8_e4m3fn:
            fully_shard(layer.mlp.experts, mesh=mesh, mp_policy=MixedPrecisionPolicy(), reshard_after_forward=True)
        fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    visual = set(model.model.visual.parameters())
    lora = set(vision_lora)
    if not lora <= visual or any(p.requires_grad != (p in lora) for p in visual):
        raise RuntimeError('the ViT must stay frozen apart from the vision LoRA')
    model.model.visual.to(device=device)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True, ignored_params=visual)
    for buffer in model.buffers():
        if buffer.device.type != 'cuda':
            buffer.data = buffer.data.to(device)
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


def sharded_parameters(model, names, before, replicated=()):
    """The trainable Parameters after shard(), by name: FSDP2 swapped them for DTensor Parameters, so ``before`` (the
    pre-FSDP objects) must not reach the optimizer. Raises unless every one is a new, trainable FP32 DTensor.
    ``replicated``: the other trainable parameters (shard's ``vision_lora``), which FSDP left as they were."""
    from torch.distributed.tensor import DTensor

    params = dict(model.named_parameters())
    after = [params[n] for n in names]
    stale = [n for n, a, b in zip(names, after, before, strict=True) if a is b]
    wrong = [n for n, p in zip(names, after) if not (isinstance(p, DTensor) and p.requires_grad and p.dtype == torch.float32)]
    if stale or wrong:
        raise RuntimeError(f'after FSDP: {len(stale)} trainable parameters not replaced ({stale[:3]}), {len(wrong)} not '
                           f'trainable FP32 DTensors ({wrong[:3]})')
    own = {id(p) for p in model.parameters()}
    if not all(id(p) in own and p.requires_grad and not isinstance(p, DTensor) for p in replicated):
        raise RuntimeError('a replicated trainable parameter is no longer a plain trainable parameter of the model')
    if sum(p.requires_grad for p in model.parameters()) != len(after) + len(replicated):
        raise RuntimeError('the sharded model has trainable parameters outside the trainable set')
    return after


def full_tensors(names, parameters, rank):
    """{name: full BF16 CPU tensor} of the sharded masters on rank 0 (None elsewhere); every rank joins the gathers.
    BF16 = the cast FSDP hands the forward, i.e. what the server must run."""
    out = {} if rank == 0 else None
    with torch.no_grad():
        for name, p in zip(names, parameters, strict=True):
            full = p.full_tensor() if hasattr(p, 'full_tensor') else p.detach()
            if rank == 0:
                out[name] = full.to(torch.bfloat16).cpu()
            del full
    return out


#: top-level fields of a re-export that must equal the published manifest's (the rest are timings and provenance)
RESUME_EXPORT_FIELDS = ('format', 'weight_version', 'base_model', 'tensors', 'bytes', 'trainer', 'trained_keys')


def export_differences(published, again):
    """What differs between the published ``merged_manifest.json`` of a version and a re-export of the restored masters
    at that version: the RESUME_EXPORT_FIELDS and every key's entry (shape, format, base / merged sha256, changed
    elements). [] = the same bytes."""
    diff = [f'{f}: {published.get(f)!r} != {again.get(f)!r}' for f in RESUME_EXPORT_FIELDS
            if published.get(f) != again.get(f)]
    a, b = published['keys'], again['keys']
    if set(a) != set(b):
        diff.append(f'keys: {len(set(a) - set(b))} only published (e.g. {sorted(set(a) - set(b))[:2]}), '
                    f'{len(set(b) - set(a))} only re-exported (e.g. {sorted(set(b) - set(a))[:2]})')
    changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    if changed:
        diff.append(f'{len(changed)} tensors differ, e.g. {changed[:3]}: {a[changed[0]]} != {b[changed[0]]}')
    return diff


def check_resumed_export(names, parameters, rank, exporter, out, weight_version, scratch, group):
    """Resume gate (user 10-30, the first full-weight resume): the restored masters exported again at the checkpoint's
    weight version must be the published ``<out>/merged/v<weight_version>`` byte for byte (export_differences). Every
    rank joins the gathers; rank 0 writes the re-export into a temporary directory under ``scratch``, compares, removes
    it, and the verdict goes to every rank over ``group`` so all of them raise together."""
    import json
    import shutil
    import tempfile

    import torch.distributed as dist

    weights = full_tensors(names, parameters, rank)
    verdict = [None]
    if rank == 0:
        published = Path(out) / 'merged' / f'v{weight_version:06d}'
        tmp = Path(tempfile.mkdtemp(prefix=f'resume-check-v{weight_version:06d}-', dir=scratch))
        try:
            if not (published / 'DONE').is_file():
                raise RuntimeError(f'{published}: no DONE, nothing to compare the restored weights with')
            again = exporter.write_full(tmp / published.name, weights, weight_version)
            diff = export_differences(json.loads((published / 'merged_manifest.json').read_text()), again)
            if diff:
                verdict[0] = f'restored weights differ from {published}: ' + '; '.join(diff)
            else:
                print(f'[rank0] resume gate: the restored masters re-export {published.name} byte for byte '
                      f'({again["tensors"]} tensors, {again["bytes"]} bytes)', flush=True)
        except Exception as error:   # handed to every rank below, which all raise
            verdict[0] = f'{type(error).__name__}: {error}'
        finally:
            shutil.rmtree(tmp)
    del weights
    dist.broadcast_object_list(verdict, src=0, group=group)
    if verdict[0] is not None:
        raise RuntimeError(f'resume gate: {verdict[0]}')


def assert_gradients(names, parameters):
    """Every trainable master holds a gradient (raises with the names otherwise, before any optimizer state changes)."""
    missing = [n for n, p in zip(names, parameters, strict=True) if p.grad is None]
    if missing:
        layers = sorted({layer_of(n) for n in missing})
        raise RuntimeError(f'{len(missing)} of {len(names)} trainable tensors got no gradient (layers {layers}): '
                           f'{missing[:6]}')


def clip_gradients(names, parameters, max_norm, replicated=()):
    """Global gradient norm of the sharded (already summed over ranks) gradients, then clip; raises on every rank
    before any optimizer state changes if a gradient is missing or the norm is not finite and positive.
    ``replicated``: plain parameters whose gradients are already the same on every rank (the vision LoRA after
    sync_gradients); their squares join the global norm and they are clipped with the same coefficient as
    clip_grad_norm_ uses (max_norm / (norm + 1e-6), at most 1)."""
    assert_gradients(names, parameters)
    if not replicated:
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=False)
        norm = float(norm.full_tensor() if hasattr(norm, 'full_tensor') else norm)
        if not (norm > 0 and norm < float('inf')):
            raise RuntimeError(f'gradient norm {norm}')
        return norm
    import torch.distributed as dist

    missing = [i for i, p in enumerate(replicated) if p.grad is None]
    if missing:
        raise RuntimeError(f'{len(missing)} replicated trainable tensors got no gradient')
    device = replicated[0].grad.device
    sharded = torch.zeros((), device=device, dtype=torch.float32)
    for p in parameters:   # each rank holds disjoint dim-0 shards (to_local drops FSDP2's padding)
        sharded += torch.linalg.vector_norm(p.grad.to_local().float()) ** 2
    dist.all_reduce(sharded)
    local = sum(torch.linalg.vector_norm(p.grad.float()) ** 2 for p in replicated)
    norm = float(torch.sqrt(sharded + local))
    if not (norm > 0 and norm < float('inf')):
        raise RuntimeError(f'gradient norm {norm}')
    coef = max_norm / (norm + 1e-6)
    if coef < 1.0:
        with torch.no_grad():
            for p in (*parameters, *replicated):
                p.grad.mul_(coef)
    return norm


def checkpoint_state(model, optimizer, names):
    """{'model': trainable masters, 'optim': AdamW state} for torch.distributed.checkpoint (train_qad.ckpt_state)."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    msd, osd = get_state_dict(model, optimizer, options=StateDictOptions(ignore_frozen_params=True))
    buffers = {n for n, _ in model.named_buffers()}
    extra = set(msd) - set(names)
    if set(names) - set(msd) or extra - buffers:
        raise RuntimeError(f'checkpoint misses {sorted(set(names) - set(msd))[:3]} or holds non-buffer extras '
                           f'{sorted(extra - buffers)[:3]}')
    return {'model': {n: msd[n] for n in names}, 'optim': osd}


def save(model, optimizer, names, path):
    import torch.distributed.checkpoint as dcp

    dcp.save(checkpoint_state(model, optimizer, names), checkpoint_id=str(path))


def restore(model, optimizer, names, path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict

    state = checkpoint_state(model, optimizer, names)
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(model, optimizer, model_state_dict=state['model'], optim_state_dict=state['optim'],
                   options=StateDictOptions(ignore_frozen_params=True, strict=False))
