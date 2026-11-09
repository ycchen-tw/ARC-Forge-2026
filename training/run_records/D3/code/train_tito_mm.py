"""LoRA training of Qwen3.8-Flash-Next on TITO segments: <=120k packed bins with images and video.

One bin of whole segments per rank and step (no truncation). Frozen base (FP8
experts or BF16), frozen indexer with real QSA routes, frozen ViT (``--vit-lora-rank N``: plus a
LoRA on its projections, trained with the text LoRA; the ViT base weights stay frozen). Vision
embeddings are computed per media item with SGLang's position interpolation
(linspace, bf16 weights; vision_interp.py) and written into the ledger's pad positions;
M-RoPE positions come from the data (SGLang's computation). The loss is the
token-weighted CE over sampled (gen) tokens of all ranks. Checkpointed residual
inputs can live in host memory (checkpoint_offload.py).

Modes: ``train`` walks shuffled balanced steps (tito_bins.py); ``bench`` replays
a fixed list of steps with LR=0 for systems comparisons; ``score`` writes per-token
log-probabilities of chosen segments.

RL plumbing: ``--routing-replay force`` makes every MoE layer take the experts SGLang chose
(routing_replay.py; data from ``prepare_tito_mm.py --routing``), ``observe`` only measures
agreement; ``--save-merged-every`` publishes the LoRA-merged projections for SGLang's
update_weights_from_disk (merged_export.py). ``--dense-qat fp8 --dense-qat-reference <deployed package>`` trains the
projections the deployed model runs as block FP8 through quantize(x) @ quantize(W + LoRA)^T (fp8_dense_qat.py) and
exports them as FP8 + scale_inv (merged format v2).
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import fp8_dense_qat
import numpy as np
import torch
import torch.distributed as dist
import trainer_kernels

QSA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
GDN_TARGETS = ('in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj')
# Every layer's shared expert (BF16 in the deployed packages; its sigmoid gate and the router are not adapted).
SHARED_EXPERT_TARGETS = ('gate_proj', 'up_proj', 'down_proj')

# --vit-lora-rank > 0: LoRA on these ViT projections (names inside model.visual); base ViT weights stay frozen.
VIT_TARGETS = {'blocks': ('attn.qkv', 'attn.proj', 'mlp.linear_fc1', 'mlp.linear_fc2'),  # per block (27 in FN)
               'merger': ('linear_fc1', 'linear_fc2')}
VIT_PREFIX = 'base_model.model.model.visual.'  # PEFT parameter names of the ViT
# adapter_config.json pattern that gives the vision LoRA its own rank / alpha (PEFT matches it to the full module name)
VIT_PATTERN = r'model\.visual\..*'
AUX_HEAD_PREFIX = 'vit_aux_color_head.'  # names of the --vit-aux-color-loss head (trainer only; merged_export skips it)
IGNORE = -100
# --vit-aux-color-loss: per merger token, the colour of each 16x16 quadrant's centre pixel -- ARC palette index 0-9,
# COLOR_PAD for the white padding of the 16pad renderer (16 px per cell, one token = 2x2 cells)
COLOR_PAD = 10
COLOR_CLASSES = 11
_ARC_RGB = []  # the ARC palette, loaded once from work/arc_code/arc2/arc2_render.py


def arc_palette():
    """(10, 3) uint8 RGB of the ARC colours, as the renderer draws them (arc2_render.RGB)."""
    if not _ARC_RGB:
        import importlib.util
        path = Path(__file__).resolve().parents[1] / 'arc_code' / 'arc2' / 'arc2_render.py'
        spec = importlib.util.spec_from_file_location('arc2_render', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.RGB.shape != (10, 3) or module.RGB.dtype != np.uint8 or (module.RGB == 255).all(-1).any():
            raise RuntimeError(f'{path}: unexpected ARC palette {module.RGB.tolist()}')
        _ARC_RGB.append(torch.from_numpy(module.RGB.astype(np.int64)))
    return _ARC_RGB[0]


def quadrant_color_labels(pixels, grid_thw, vision_config):
    """(h*w/4, 4) int64 colour classes of one image's merger tokens (token order = the processor's, row-major over
    the 2x2 merge windows; quadrants (0,0), (0,1), (1,0), (1,1)), from its patchified float32 pixels
    (tito_mm_data.patchify: rows = patches with each merge window's four patches consecutive, columns = (channel,
    temporal copy, y, x); values (u8 - 127.5) / 127.5). With patch 16 and merge 2 a quadrant is exactly one patch, so
    its label is the patch's centre pixel (8, 8): an ARC palette colour -> 0-9, white -> COLOR_PAD; any other colour
    raises (not a 16pad grid image)."""
    patch, temporal, merge = vision_config.patch_size, vision_config.temporal_patch_size, vision_config.spatial_merge_size
    if (patch, merge) != (16, 2):
        raise ValueError(f'colour labels need patch 16 / merge 2 (one patch per quadrant), not {patch} / {merge}')
    t, h, w = grid_thw
    if t != 1:
        raise ValueError(f'colour labels are for images; grid {grid_thw}')
    if pixels.dtype != torch.float32 or tuple(pixels.shape) != (h * w, 3 * temporal * patch * patch):
        raise ValueError(f'pixels {pixels.dtype} {tuple(pixels.shape)} for grid {grid_thw}')
    centre = pixels.view(h * w, 3, temporal, patch, patch)[:, :, 0, patch // 2, patch // 2]
    u8 = centre * 127.5 + 127.5
    rgb = u8.round()
    if (u8 - rgb).abs().max() > 1e-3 or rgb.min() < 0 or rgb.max() > 255:
        raise ValueError('pixels are not the processor normalization of uint8 frames')
    rgb = rgb.to(torch.int64)
    palette = arc_palette()
    match = (rgb[:, None, :] == palette[None]).all(-1)
    white = (rgb == 255).all(-1)
    unknown = ~(match.any(-1) | white)
    if unknown.any():
        first = int(unknown.nonzero()[0])
        raise ValueError(f'{int(unknown.sum())} of {len(rgb)} quadrant centres are neither an ARC colour nor white '
                         f'(patch {first}: RGB {rgb[first].tolist()}): not a 16pad grid image')
    labels = torch.where(white, COLOR_PAD, match.to(torch.int64).argmax(-1))
    return labels.view(h * w // 4, 4)


def tito_forward(self, input_ids=None, labels=None, position_ids=None, packed_meta=None, media=None,
                 attention_mask=None, inputs_embeds=None, score=False, ple_input_ids=None, temperature=1.0,
                 aux_color_head=None, **kwargs):
    """Embeddings -> frozen ViT scatter -> packed text model -> summed CE on supervised positions.

    ``score=True`` instead returns the fp32 log-probability of every label (no reduction) under
    ``logits / temperature`` (the sampling distribution the TITO ledger's logprobs are taken from).
    ``aux_color_head`` (--vit-aux-color-loss): also the summed quadrant-colour CE of the merger output against
    ``media['color_labels']`` (color_aux_terms); the result is then a dict with ``loss`` and those terms.
    """
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    from transformers.modeling_outputs import CausalLMOutputWithPast

    vision_out = None if aux_color_head is None else []
    shifted, wanted, valid = supervised_hidden(self, input_ids, labels, position_ids, packed_meta, media,
                                               attention_mask, inputs_embeds, ple_input_ids, kwargs,
                                               vision_out=vision_out)
    if score:
        if aux_color_head is not None:
            raise ValueError('score mode has no auxiliary loss')
        parts = []
        for start in range(0, len(wanted), 4096):
            logits = torch.nn.functional.linear(shifted[start:start + 4096], self.lm_head.weight).float() / temperature
            parts.append(logits.log_softmax(-1).gather(1, wanted[start:start + 4096, None])[:, 0])
        return {'token_logprobs': torch.cat(parts), 'valid': valid}
    loss_sum = LigerFusedLinearCrossEntropyLoss(reduction='sum')(self.lm_head.weight, shifted, wanted)
    if aux_color_head is None:
        return CausalLMOutputWithPast(loss=loss_sum)
    labels = None if media is None else media['color_labels']
    return {'loss': loss_sum, **color_aux_terms(aux_color_head, vision_out[0], labels, loss_sum)}


def color_aux_terms(head, vision, labels, like):
    """--vit-aux-color-loss on one rank: ``head`` (Linear(out_hidden, 4 * COLOR_CLASSES)) reads each merger token's
    four 16x16 quadrants. Sums, not means (the caller normalizes by the global quadrant count): ``aux_sum`` (CE, carries
    the gradient), ``aux_correct`` / ``aux_count`` over all quadrants, ``aux_cell_correct`` / ``aux_cells`` over the
    quadrants on grid cells (label != COLOR_PAD). No media on this rank (``vision`` None): zeros like ``like``."""
    if vision is None:
        if labels is not None:
            raise RuntimeError('colour labels without vision tokens')
        zero = torch.zeros((), dtype=torch.float32, device=like.device)
        return {'aux_sum': zero, 'aux_correct': zero, 'aux_count': zero, 'aux_cell_correct': zero, 'aux_cells': zero}
    if labels is None or labels.shape != (vision.shape[0], 4):
        raise RuntimeError(f'colour labels {None if labels is None else tuple(labels.shape)} for '
                           f'{vision.shape[0]} vision tokens')
    logits = head(vision.float()).view(-1, COLOR_CLASSES)
    flat = labels.reshape(-1)
    aux_sum = torch.nn.functional.cross_entropy(logits, flat, reduction='sum')
    with torch.no_grad():
        hit = logits.argmax(-1) == flat
        cell = flat != COLOR_PAD
        terms = [hit.sum(), torch.tensor(flat.numel(), device=flat.device), (hit & cell).sum(), cell.sum()]
    return {'aux_sum': aux_sum, **dict(zip(('aux_correct', 'aux_count', 'aux_cell_correct', 'aux_cells'),
                                           (t.float() for t in terms), strict=True))}


def supervised_hidden(self, input_ids, labels, position_ids, packed_meta, media, attention_mask, inputs_embeds,
                      ple_input_ids, kwargs, vision_out=None, supervised=None):
    """The forward shared by tito_forward and train_rl_service.rl_forward: final hidden states at the positions that
    predict a supervised label, in pack order, with those labels and the (T-1,) mask of supervised next tokens.
    ``vision_out`` (a list): the merger output (the vision tokens written into the embeddings; None without media) is
    appended to it. ``supervised``: None selects the positions with the boolean mask (its count is a host sync that
    drains the GPU queue at the end of the forward); the caller's host-side count of supervised labels selects them with
    nonzero_static instead, the same rows in the same order without a sync -- the caller then checks
    ``valid.sum() == supervised`` after its own synchronize (too few found -> index -1 -> device assert; too many ->
    rows dropped, which that check raises on)."""
    if inputs_embeds is not None or attention_mask is None or input_ids is None or labels is None:
        raise ValueError('expects ids, labels and the guarded packed mask dictionary')
    if ple_input_ids is None or ple_input_ids.shape != input_ids.shape:
        raise ValueError('PLE ids must be given explicitly (see build_batch)')
    passthrough = {k: kwargs.pop(k) for k in ('cu_seq_lens_q', 'cu_seq_lens_k', 'max_length_q', 'max_length_k',
                                              'seq_idx')}
    stray = {k: v for k, v in kwargs.items() if v not in (None, False)}
    if stray:
        raise ValueError(f'unexpected forward arguments: {sorted(stray)}')
    text = self.model.language_model
    embeds = text.embed_tokens(input_ids)
    if media is not None:
        interp = {} if media['interp_indices'] is None else {'interp_indices': media['interp_indices'],
                                                             'interp_weights': media['interp_weights']}
        # set by load_trainer: with a vision LoRA the ViT runs under the caller's grad mode, else always frozen
        with nullcontext() if self.vit_lora else torch.no_grad():
            vision = self.model.visual(media['pixels'], grid_thw=media['grid_thw'], **interp).pooler_output
        if vision.shape[0] != media['positions'].numel():
            raise RuntimeError(f'{vision.shape[0]} vision tokens for {media["positions"].numel()} pad positions')
        embeds = embeds.clone()
        embeds[0, media['positions']] = vision.to(embeds.dtype)
        if vision_out is not None:
            vision_out.append(vision)
    elif vision_out is not None:
        vision_out.append(None)
    hidden = text(inputs_embeds=embeds, ple_input_ids=ple_input_ids, position_ids=position_ids,
                  attention_mask=attention_mask, use_cache=False, packed_meta=packed_meta,
                  **passthrough).last_hidden_state
    target = labels[:, 1:].reshape(-1)
    valid = target != IGNORE
    if supervised is None:
        shifted = hidden[:, :-1].reshape(-1, hidden.shape[-1])[valid].contiguous()
        return shifted, target[valid], valid
    index = torch.nonzero_static(valid, size=supervised).squeeze(1)
    return hidden[:, :-1].reshape(-1, hidden.shape[-1]).index_select(0, index), target.index_select(0, index), valid


def qsa_backend(choice, rows):
    """'auto' = the gather kernel (qsa_gather.py: reads only each query's selected keys, packed bins).
    'flex' = the fastest Flex mask the bin allows (single_tma needs one segment and T % 128 == 0)."""
    if choice == 'auto':
        return 'gather'
    if choice != 'flex':
        return choice
    if len(rows) > 1:
        return 'generic'
    return 'single_tma' if len(rows[0]['input_ids']) % 128 == 0 else 'single'


# Arguments a resumed run may change: bookkeeping, debug checks and inputs of the other modes. Everything else
# shapes what is trained (LambdaLR does not restore its lambda's warmup or floor; alpha, data and backends live in
# no saved state), so a resume must repeat it exactly.
RESUME_MAY_DIFFER = frozenset({
    'out', 'resume', 'eval_every', 'eval_steps', 'save_every', 'audit_steps', 'offload_checkpoint_inputs',
    'offload_mode', 'offload_keep_gib', 'verify_offload', 'dump_grads', 'ple_crosscheck', 'recompute_probe_layer',
    'score_keys', 'score_variants', 'bench_steps', 'warmup', 'repeats', 'score_temperature', 'save_merged_every',
    'init_lora', 'dense_qat_reference', 'dense_qat_check_export'})  # a resume restores the LoRA from the checkpoint;
# --init-lora only seeds a fresh run; the QAT reference package only verifies the base (--dense-qat itself shapes training)
# Training settings added after runs had been saved: a checkpoint without the key was trained with this value.
ADDED_SETTINGS = {'routing_replay': 'none', 'ple_codec': 'fp8', 'dense_qat': 'none', 'fast_kernels': 'none',
                  'lora_shared_expert': False, 'int3_act': 'fp8', 'dense_qat_gemm': 'triton',
                  'vit_lora_rank': 0, 'vit_lora_alpha': 16, 'vit_lora_targets': 'blocks,merger', 'vit_lr': None,
                  'vit_lora_sdpa': 'auto', 'vit_aux_color_loss': 0.0}


def resume_conflicts(saved, current, may_differ=RESUME_MAY_DIFFER, added=ADDED_SETTINGS):
    """{argument: (saved, current)} for every training setting a resume would silently change."""
    saved, current = {**added, **saved}, {**added, **current}
    absent = object()
    keys = (saved.keys() | current.keys()) - may_differ
    return {k: (saved.get(k, '<absent>'), current.get(k, '<absent>')) for k in sorted(keys)
            if saved.get(k, absent) != current.get(k, absent)}


def host_rss_gib():
    """Resident set of this rank (pinned offload buffers included), for soak tests."""
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) / 2**20
    raise RuntimeError('no VmRSS in /proc/self/status')


def build_batch(rows, pad_token, device, vision_config, ids_bound, score=False, ple_media='deployed',
                vit_interp='deployed', audit=True, color_labels=False):
    """Pack rows (dataset items) into one B=1 sequence plus media tensors in pack order.

    ``ple_media='deployed'``: the PLE n-gram ids read ``ids_bound - 1`` at every media pad, as the deployed
    SGLang does (its multimodal embedding clamps input_ids in place; asserted by
    work/qwen38fn_dflash/capture_tito.py check_capture) and as the TITO data were generated.
    ``'semantic'`` keeps the <|image_pad|>/<|video_pad|> ids (HF / old patch 0020 semantics).
    ``vit_interp='deployed'`` passes SGLang's interpolation (linspace, bf16 weights, vision_interp.py);
    ``'hf'`` leaves HF's own (same geometry, fp32 weights) -- a control, not a training mode.
    ``color_labels`` (--vit-aux-color-loss): ``media['color_labels']`` = quadrant_color_labels of every image, in
    vision-token order.
    """
    from packed_qwen4 import RuntimeRecords, pack_segments
    from vision_interp import sglang_vit_interp

    segments = []
    for row in rows:
        ids = row['input_ids']
        if ids.dtype != np.int64 or row['loss_mask'].dtype != np.bool_ or len(ids) != len(row['loss_mask']):
            raise TypeError('dataset rows must carry int64 ids and a bool loss mask of the same length')
        if ids.min() < 0 or ids.max() >= ids_bound:
            raise ValueError(f'token id outside the model vocabulary in {row["key"]}')
        segments.append((ids, ids if score else np.where(row['loss_mask'], ids, IGNORE)))
    batch = pack_segments(segments, pad_token, device)
    total = batch['input_ids'].shape[1]
    text_positions = batch['position_ids'][:1]  # (1, 1, T): per-segment arange, the text axis
    mrope = torch.from_numpy(np.concatenate([row['mrope_positions'] for row in rows], axis=1)).to(device)
    if mrope.shape != (3, total):
        raise ValueError(f'mrope positions {tuple(mrope.shape)} for {total} tokens')
    batch['position_ids'] = torch.cat([text_positions, mrope[:, None, :].to(text_positions.dtype)], dim=0)
    pixels, grids, positions, colors = [], [], [], []
    offset = 0
    for row in rows:
        for item in row['media']:
            t, h, w = item['grid_thw']
            span = np.concatenate([np.arange(a, b + 1) for a, b in item['ranges']])
            if len(span) != t * h * w // (vision_config.spatial_merge_size ** 2):
                raise ValueError(f'{row["key"]}: {len(span)} pad positions for grid {(t, h, w)}')
            if item['pixels'].shape[0] != t * h * w:
                raise ValueError(f'{row["key"]}: {item["pixels"].shape[0]} pixel rows for grid {(t, h, w)}')
            pixels.append(item['pixels'])
            if color_labels:  # from the float32 pixels, before the bf16 cast
                colors.append(quadrant_color_labels(item['pixels'], (t, h, w), vision_config))
            grids.append((t, h, w))
            positions.append(span + offset)
        offset += len(row['input_ids'])
    meta = batch['packed_meta']
    if audit:  # per-layer selection coverage check; costs host syncs, so only on audited steps
        meta['qsa_audit'] = RuntimeRecords()
    media = None
    if pixels:
        side = int(vision_config.num_position_embeddings ** 0.5)
        if vit_interp == 'deployed':
            indices, weights = sglang_vit_interp(grids, side, vision_config.spatial_merge_size, torch.bfloat16,
                                                 device)
        elif vit_interp == 'hf':
            indices = weights = None
        else:
            raise ValueError(f'unknown ViT interpolation {vit_interp!r}')
        media = {'pixels': torch.cat(pixels).to(device=device, dtype=torch.bfloat16, non_blocking=True),
                 'grid_thw': torch.tensor(grids, dtype=torch.long, device=device),
                 'interp_indices': indices, 'interp_weights': weights,
                 'positions': torch.from_numpy(np.concatenate(positions)).to(device)}
        if color_labels:
            media['color_labels'] = torch.cat(colors).to(device)
        pad_ids = batch['input_ids'][0, media['positions']]
        if not torch.isin(pad_ids, torch.tensor(vision_config.pad_ids, device=device)).all():
            raise ValueError('vision tokens would overwrite non-pad ids')
    on_pad = torch.isin(batch['input_ids'], torch.tensor(vision_config.pad_ids, device=device))
    if int(on_pad.sum()) != (0 if media is None else media['positions'].numel()):
        raise ValueError('media pad ids outside the declared media ranges')
    if ple_media == 'deployed':
        batch['ple_input_ids'] = torch.where(on_pad, ids_bound - 1, batch['input_ids'])
    elif ple_media == 'semantic':
        batch['ple_input_ids'] = batch['input_ids']
    else:
        raise ValueError(f'unknown PLE media id mode {ple_media!r}')
    return batch, media


def lora_targets(model, shared_expert):
    """Module names the LoRA adapts: the 228 QSA/GDN projections, plus (``shared_expert``) the 144 shared-expert
    gate/up/down projections of the 48 layers (372)."""
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.startswith('model.language_model.layers.'):
            continue
        parent, leaf = name.rsplit('.', 1)
        if (leaf in QSA_TARGETS and '.self_attn.' in name) or (leaf in GDN_TARGETS and '.linear_attn.' in name) \
                or (shared_expert and leaf in SHARED_EXPERT_TARGETS and parent.endswith('.mlp.shared_expert')):
            names.append(name)
    expected = 12 * len(QSA_TARGETS) + 36 * len(GDN_TARGETS) + (48 * len(SHARED_EXPERT_TARGETS) if shared_expert else 0)
    if len(names) != expected:
        raise RuntimeError(f'expected {expected} LoRA projections (shared expert: {shared_expert}), found {len(names)}')
    return names


def add_lora_target_arguments(parser):
    parser.add_argument('--lora-shared-expert', action=argparse.BooleanOptionalAction, default=True,
                        help='also adapt every layer\'s shared expert gate/up/down_proj (BF16, merged and published as '
                             'BF16); --no-lora-shared-expert = the 228 attention projections only, as runs saved '
                             'before the flag')


def vit_lora_targets(visual, groups):
    """Names inside the ViT (``blocks.N.attn.qkv``, ``merger.linear_fc1``, ...) of the projections ``groups`` (a
    subset of VIT_TARGETS' keys) adapts, in module order."""
    if not groups or len(set(groups)) != len(groups) or not set(groups) <= set(VIT_TARGETS):
        raise ValueError(f'ViT LoRA target groups {groups}: a non-empty subset of {sorted(VIT_TARGETS)}')
    names = []
    if 'blocks' in groups:
        names += [f'blocks.{i}.{leaf}' for i in range(len(visual.blocks)) for leaf in VIT_TARGETS['blocks']]
    if 'merger' in groups:
        names += [f'merger.{leaf}' for leaf in VIT_TARGETS['merger']]
    for name in names:
        kind = type(visual.get_submodule(name))
        if kind is not torch.nn.Linear:
            raise RuntimeError(f'ViT {name} is {kind.__module__}.{kind.__qualname__}, not a plain torch Linear')
    return names


def attach_vision_lora(visual, rank, alpha, groups, device, broadcast):
    """Wrap the ViT projections of ``groups`` in PEFT lora.Linear (adapter ``default``, dropout 0, PEFT's init: A
    Kaiming-uniform, B = 0, so the ViT's output is unchanged) with fp32 A/B on ``device``; the frozen base weights stay
    where they are. ``broadcast``: rank 0's initialization is copied to every rank (as attach_replicated_lora does).
    Returns [(name inside the ViT, parameter)] of the new LoRA weights in module order (A before B per module)."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    targets = vit_lora_targets(visual, groups)
    if any(p.requires_grad for p in visual.parameters()):
        raise RuntimeError('the ViT base weights must be frozen before the vision LoRA is attached')
    for name in targets:
        parent, leaf = name.rsplit('.', 1)
        owner = visual.get_submodule(parent)
        setattr(owner, leaf, LoraLinear(getattr(owner, leaf), 'default', r=rank, lora_alpha=alpha, lora_dropout=0.0,
                                        init_lora_weights=True))
    trainable = [(n, p) for n, p in visual.named_parameters() if p.requires_grad]
    expected = [f'{t}.lora_{part}.default.weight' for t in targets for part in 'AB']
    if [n for n, _ in trainable] != expected:
        raise RuntimeError(f'unexpected trainable ViT parameters: {[n for n, _ in trainable][:6]}')
    for _, p in trainable:
        p.data = p.data.to(device=device, dtype=torch.float32)
        if broadcast:
            dist.broadcast(p.data, 0)
    if any(bool(p.detach().any()) for n, p in trainable if '.lora_B.' in n):
        raise RuntimeError('vision LoRA B must start at zero')
    return trainable


def load_trainer(args, rank, device, world, routing_mode, merged_export, shadow_kernels=False):
    """Model setup shared with train_rl_service.py: frozen multimodal base, replicated LoRA (optionally seeded from
    ``args.init_lora``), FSDP, ``tito_forward``, FlashQLA call counter, FP8 dX mode, MoE routing replay
    (``routing_mode`` 'none' | 'observe' | 'force'), on rank 0 a MergedExporter if ``merged_export``, and the pinned
    host slots of the checkpoint offload. With ``args.dense_qat == 'fp8'`` the package's block-FP8 projections run
    through fp8_dense_qat (base verified against ``args.dense_qat_reference``) and the exporter writes merged v2.
    ``args.fast_kernels`` binds trainer_kernels (``shadow_kernels``: each call also runs the replaced forward).
    ``args.vit_lora_rank`` > 0 adds a vision LoRA (attach_vision_lora; ``vision_names`` / ``vision_parameters``, the
    tail of ``names`` / ``parameters``) that is replicated, synchronized, saved and exported with the text LoRA.
    Reads args.model, base_storage, ple_codec, lora_rank, lora_alpha, lora_shared_expert, seed, init_lora, fp8_dx,
    offload_checkpoint_inputs, bin_tokens, dense_qat, dense_qat_reference, fast_kernels, vit_lora_* (absent = none).
    ``args.trainable == 'full'`` (train_rl_service.py): no LoRA; the same projections train as FP32 full weights
    (dense_full_rl.py; ``full`` in the result, ``base`` is the model itself, the exporter is in full mode)."""
    from probe_train import import_stack
    _, _, lora_cls, task_type, get_peft = import_stack()
    from checkpoint_offload import PinnedSlots
    from distributed_utils import observe_flash_qla
    from multimodal import load_multimodal, shard_multimodal
    from torch.distributed.device_mesh import init_device_mesh
    from train_viggo_fsdp import attach_replicated_lora

    torch.manual_seed(args.seed)
    print(f'[rank{rank}] loading {args.base_storage}', flush=True)
    # Saved args of runs from before --ple-codec existed (probes replay them) are FP8 PLE runs.
    ple_codec = getattr(args, 'ple_codec', ADDED_SETTINGS['ple_codec'])
    model, loading, kernels = load_multimodal(args.model, args.base_storage, device, ple_codec)
    # Saved args of runs from before --lora-shared-expert existed (probes replay them) adapted attention only.
    names = lora_targets(model, getattr(args, 'lora_shared_expert', ADDED_SETTINGS['lora_shared_expert']))
    if getattr(args, 'trainable', 'lora') == 'full':
        return load_full_trainer(args, rank, device, world, routing_mode, merged_export, shadow_kernels, model, names,
                                 loading, kernels)
    model, parameters, names = attach_replicated_lora(
        model, lora_cls, task_type, get_peft,
        SimpleNamespace(rank=args.lora_rank, alpha=args.lora_alpha, targets=names), device)
    dense_qat = None
    if args.dense_qat == 'fp8':  # before FSDP: the BF16 base is still whole on the CPU for the reference check
        # saved args from before --dense-qat-gemm ran the Triton GEMM
        gemm = getattr(args, 'dense_qat_gemm', ADDED_SETTINGS['dense_qat_gemm'])
        fp8_dense_qat.set_gemm(gemm, device)
        dense_qat = fp8_dense_qat.install(model, args.dense_qat_reference, device, args.lora_alpha / args.lora_rank)
        print(f'[rank{rank}] dense QAT: {len(dense_qat["modules"])} block-FP8 projections ({gemm} GEMM), base verified '
              f'bit for bit against {args.dense_qat_reference}: {dense_qat["verified"]}', flush=True)
    # Saved args of runs from before --fast-kernels existed (probes replay them) ran the reference kernels.
    fast_kernels = trainer_kernels.parse(getattr(args, 'fast_kernels', ADDED_SETTINGS['fast_kernels']))
    if fast_kernels:  # after dense QAT bound its projections (they keep their own forward), before FSDP
        bound = trainer_kernels.install(model, fast_kernels, shadow=shadow_kernels,
                                        skip_lora=() if dense_qat is None else dense_qat['modules'].values())
        print(f'[rank{rank}] fast kernels: {bound}{" (shadowed)" if shadow_kernels else ""}', flush=True)
    elif shadow_kernels:
        raise ValueError('shadow_kernels without --fast-kernels')
    base = model.base_model.model
    # After the text LoRA (its initialization draws from the RNG first, so it is the same with or without a vision
    # LoRA), dense QAT and the fast kernels (they bind text projections only).
    vit_rank, vit_alpha, vit_groups, vit_sdpa = vit_lora_settings(args)
    text_names = list(names)
    vision_names = []
    if vit_rank:
        attached = attach_vision_lora(base.model.visual, vit_rank, vit_alpha, vit_groups, device, broadcast=True)
        use_vit_sdpa(base.model.visual, vit_sdpa)
        vision_names = [VIT_PREFIX + n for n, _ in attached]
        names, parameters = names + vision_names, parameters + [p for _, p in attached]
        config = model.peft_config['default']  # what checkpoint-<step>/adapter_config.json records
        config.target_modules = set(config.target_modules) | {
            'model.visual.' + n.removesuffix('.lora_A.default.weight') for n, _ in attached if '.lora_A.' in n}
        if (vit_rank, vit_alpha) != (args.lora_rank, args.lora_alpha):
            config.rank_pattern = {**config.rank_pattern, VIT_PATTERN: vit_rank}
            config.alpha_pattern = {**config.alpha_pattern, VIT_PATTERN: vit_alpha}
        print(f'[rank{rank}] vision LoRA r{vit_rank}/alpha {vit_alpha} on {len(vision_names) // 2} ViT projections '
              f'({",".join(vit_groups)}): {sum(p.numel() for _, p in attached)} parameters, ViT SDPA {vit_sdpa}',
              flush=True)
    aux_head = None
    if aux_color_weight(args):  # after the vision LoRA (its RNG draws), trained in the vision LoRA's param group
        out = base.config.vision_config.out_hidden_size
        aux_head = torch.nn.Linear(out, 4 * COLOR_CLASSES).to(device=device, dtype=torch.float32)
        for p in aux_head.parameters():
            dist.broadcast(p.data, 0)
        names = names + [AUX_HEAD_PREFIX + n for n, _ in aux_head.named_parameters()]
        parameters = parameters + list(aux_head.parameters())
        print(f'[rank{rank}] ViT colour head Linear({out}, 4x{COLOR_CLASSES}), loss weight {aux_color_weight(args)}',
              flush=True)
    if args.init_lora:
        state = torch.load(Path(args.init_lora) / 'trainer_state.pt', map_location='cpu', weights_only=False)
        saved = {**ADDED_SETTINGS, **state['args']}
        trained = (saved['lora_rank'], saved['lora_alpha'])
        if trained != (args.lora_rank, args.lora_alpha):
            raise RuntimeError(f'{args.init_lora}: LoRA rank/alpha {trained}, this run {(args.lora_rank, args.lora_alpha)}')
        initial = state['lora']
        del state
        if {k for k in initial if not k.startswith((VIT_PREFIX, AUX_HEAD_PREFIX))} != set(text_names):
            raise RuntimeError(f'{args.init_lora}: LoRA parameter names differ from this model (--lora-shared-expert '
                               f'must match the checkpoint)')
        saved_head = {k for k in initial if k.startswith(AUX_HEAD_PREFIX)}
        if saved_head and aux_head is None:
            print(f'[rank{rank}] --init-lora: its ViT colour head is not used by this run', flush=True)
            initial = {k: v for k, v in initial.items() if k not in saved_head}
        elif aux_head is not None and not saved_head:
            print(f'[rank{rank}] --init-lora: the ViT colour head starts fresh', flush=True)
        saved_vision = {k for k in initial if k.startswith(VIT_PREFIX)}
        if saved_vision:  # a checkpoint with a vision LoRA seeds it too; it must be this run's
            if saved_vision != set(vision_names) or (saved['vit_lora_rank'], saved['vit_lora_alpha']) != (vit_rank,
                                                                                                         vit_alpha):
                raise RuntimeError(f'{args.init_lora}: its vision LoRA (r{saved["vit_lora_rank"]}/alpha '
                                   f'{saved["vit_lora_alpha"]}, {len(saved_vision)} tensors) is not this run\'s')
        elif vision_names:
            print(f'[rank{rank}] --init-lora seeds the text LoRA; the vision LoRA starts fresh (B = 0)', flush=True)
        with torch.no_grad():
            for name, p in zip(names, parameters, strict=True):
                if name in initial:
                    p.copy_(initial[name].to(p.device))
        del initial
    trainable_vit = {VIT_PREFIX + n for n, p in base.model.visual.named_parameters() if p.requires_grad}
    if trainable_vit != set(vision_names):
        raise RuntimeError('ViT base weights must stay frozen' + (' (only the vision LoRA trains)' if vit_rank else ''))
    if any(p.requires_grad for n, p in base.named_parameters() if '.indexer.' in n):
        raise RuntimeError('indexer must stay frozen')
    base.vit_lora = bool(vision_names)  # read by supervised_hidden
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    shard_multimodal(model, parameters, init_device_mesh('cuda', (world,)), device)
    base.forward = MethodType(tito_forward, base)
    gdn = observe_flash_qla()
    if args.base_storage == 'fp8':
        from fp8_moe import set_dx_mode
        set_dx_mode(args.fp8_dx)
    elif args.fp8_dx != 'fp8':
        raise ValueError('--fp8-dx applies to the FP8 base only (the INT3 base always runs dX on dequantized BF16)')
    text_config = base.config.text_config
    # train_rl_service.py's --int3-act; this trainer and saved args from before it ran W3A8
    int3_act = getattr(args, 'int3_act', ADDED_SETTINGS['int3_act'])
    if args.base_storage == 'int3':
        import int3_moe
        int3_moe.configure(int3_act, text_config.num_hidden_layers)
        print(f'[rank{rank}] INT3 routed experts: activation {int3_act} (W3A{16 if int3_act == "bf16" else 8})',
              flush=True)
    elif int3_act != 'fp8':
        raise ValueError(f'--int3-act {int3_act} applies to the INT3 base only')
    replay = None
    if routing_mode != 'none':
        from routing_replay import RoutingReplay
        replay = RoutingReplay(routing_mode, text_config.num_hidden_layers, text_config.num_experts_per_tok)
        replay.install(base.model.language_model)
    exporter = None
    if merged_export and rank == 0:
        from merged_export import MergedExporter, adapter_tensors, lora_pairs
        adapted = sorted(lora_pairs(adapter_tensors(dict(zip(names, parameters, strict=True)))))
        exporter = MergedExporter(args.model, adapted,
                                  args.lora_rank, args.lora_alpha,
                                  dense_fp8_keys=() if dense_qat is None else [m + '.weight' for m in dense_qat['modules']],
                                  vision=(vit_rank, vit_alpha) if vision_names else None)
        print(f'[rank0] merged export: {len(exporter.base)} base projections loaded in {exporter.load_seconds:.1f} s',
              flush=True)
    vision_config = base.config.vision_config
    vision_config.pad_ids = (base.config.image_token_id, base.config.video_token_id)
    width = text_config.hidden_size * text_config.hc_count
    # Exact-size host slots for the offloaded layer inputs, shared by every step (a bin holds <= --bin-tokens).
    slots = PinnedSlots(args.bin_tokens * width) if args.offload_checkpoint_inputs else None
    return SimpleNamespace(model=model, base=base, parameters=parameters, names=names, loading=loading,
                           kernels=kernels, gdn=gdn, replay=replay, exporter=exporter, text_config=text_config,
                           vision_config=vision_config, width=width, slots=slots, dense_qat=dense_qat,
                           vision_names=vision_names, vision_parameters=parameters[len(text_names):],
                           aux_head=aux_head, full=None)


def load_full_trainer(args, rank, device, world, routing_mode, merged_export, shadow_kernels, model, targets,
                      loading, kernels):
    """load_trainer for ``--trainable full`` (dense_full_rl.py): ``targets`` (lora_targets) train as FP32 full weights,
    sharded by FSDP; the rest of the setup is load_trainer's."""
    import dense_full_rl
    from checkpoint_offload import PinnedSlots
    from distributed_utils import observe_flash_qla
    from torch.distributed.device_mesh import init_device_mesh

    if args.dense_qat != 'fp8' or args.base_storage != 'int3':
        raise ValueError('--trainable full runs the INT3 package with --dense-qat fp8 (the deployed block-FP8 projections)')
    if vit_lora_settings(args)[0] or aux_color_weight(args) or getattr(args, 'init_lora', None):
        raise ValueError('--trainable full has no vision LoRA, ViT colour head or --init-lora')
    if args.dense_qat_gemm != 'triton':
        import fp8_dense_qat as F
        F.set_gemm(args.dense_qat_gemm, device)
    full = dense_full_rl.install(model, targets, args.dense_qat_reference, device, args.freeze_first_layers,
                                 args.full_grad_kernel, args.full_grad_precision)
    print(f'[rank{rank}] full weights: {len(full.names)} of {len(full.keys)} projections trainable '
          f'({sum(p.numel() for p in full.parameters) / 1e9:.3f}B FP32), first {full.frozen["layers"]} layers frozen '
          f'{full.frozen}; {len(full.dense_fp8_keys)} block-FP8 projections ({args.dense_qat_gemm} GEMM, dW '
          f'{args.full_grad_kernel}/{args.full_grad_precision}) verified bit for bit against {args.dense_qat_reference}: '
          f'{full.verified}', flush=True)
    fast_kernels = trainer_kernels.parse(args.fast_kernels)
    dropped = [k for k in fast_kernels if k in ('lora', 'lora_grad')]
    fast_kernels = tuple(k for k in fast_kernels if k not in dropped)
    if fast_kernels:
        bound = trainer_kernels.install(model, fast_kernels, shadow=shadow_kernels)
        print(f'[rank{rank}] fast kernels: {bound} (no LoRA: {dropped or "none"} not bound)', flush=True)
    elif shadow_kernels:
        raise ValueError('shadow_kernels without --fast-kernels')
    if any(p.requires_grad for n, p in model.named_parameters() if '.indexer.' in n or n.startswith('model.visual.')):
        raise RuntimeError('indexer and ViT must stay frozen')
    model.vit_lora = False  # read by supervised_hidden
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    # gradient_checkpointing_enable also hooks the input embeddings to make their output require grad (for adapters on
    # frozen bases). Here every trained weight is a leaf of its own layer: that hook would only pull the frozen first
    # layers (--freeze-first-layers) back into the graph -- recompute and backward included -- and add a dX at layer 0.
    if not model._require_grads_hooks:
        raise RuntimeError('gradient_checkpointing_enable did not hook the input embeddings; recheck the freeze rule')
    model.disable_input_require_grads()
    if model.get_input_embeddings()._forward_hooks:
        raise RuntimeError('input embeddings still carry forward hooks after disable_input_require_grads')
    dense_full_rl.shard(model, init_device_mesh('cuda', (world,)), device)
    # fully_shard replaces every parameter with a new DTensor Parameter: the objects install() collected are gone from
    # the model (an optimizer over them would step tensors no forward reads, with no error)
    full.parameters = dense_full_rl.sharded_parameters(model, full.names, full.parameters)
    model.forward = MethodType(tito_forward, model)
    gdn = observe_flash_qla()
    text_config = model.config.text_config
    import int3_moe
    int3_moe.configure(args.int3_act, text_config.num_hidden_layers)
    print(f'[rank{rank}] INT3 routed experts: activation {args.int3_act} (W3A{16 if args.int3_act == "bf16" else 8})',
          flush=True)
    replay = None
    if routing_mode != 'none':
        from routing_replay import RoutingReplay
        replay = RoutingReplay(routing_mode, text_config.num_hidden_layers, text_config.num_experts_per_tok)
        replay.install(model.model.language_model)
    exporter = None
    if merged_export and rank == 0:
        from merged_export import MergedExporter
        exporter = MergedExporter(args.model, full.keys, None, None, dense_fp8_keys=full.dense_fp8_keys, full=True)
        print(f'[rank0] merged export (full): {len(exporter.base)} base projections loaded in '
              f'{exporter.load_seconds:.1f} s', flush=True)
    vision_config = model.config.vision_config
    vision_config.pad_ids = (model.config.image_token_id, model.config.video_token_id)
    width = text_config.hidden_size * text_config.hc_count
    slots = PinnedSlots(args.bin_tokens * width) if args.offload_checkpoint_inputs else None
    return SimpleNamespace(model=model, base=model, parameters=full.parameters, names=full.names, loading=loading,
                           kernels=kernels, gdn=gdn, replay=replay, exporter=exporter, text_config=text_config,
                           vision_config=vision_config, width=width, slots=slots, dense_qat=None,
                           vision_names=[], vision_parameters=[], aux_head=None, full=full)


def aux_color_weight(args):
    """--vit-aux-color-loss (0 = off; saved args from before it, and train_rl_service.py, have none)."""
    return getattr(args, 'vit_aux_color_loss', ADDED_SETTINGS['vit_aux_color_loss'])


def aux_summary(ranks):
    """Global --vit-aux-color-loss metrics of a step from the per-rank sums (color_aux_terms): mean CE, accuracy over
    all quadrants and over the quadrants on grid cells (white padding excluded). None where nothing was labelled."""
    total = {k: sum(r[k] for r in ranks) for k in ('aux_sum', 'aux_correct', 'aux_count', 'aux_cell_correct',
                                                    'aux_cells')}
    count, cells = total['aux_count'], total['aux_cells']
    return {'aux_color_loss': total['aux_sum'] / count if count else None,
            'aux_color_acc': total['aux_correct'] / count if count else None,
            'aux_color_acc_cells': total['aux_cell_correct'] / cells if cells else None,
            'aux_color_quadrants': int(count)}


def vit_lora_settings(args):
    """(rank, alpha, target groups, ViT SDPA choice) of the vision LoRA; rank 0 = none (saved args of runs from before
    --vit-lora-rank existed, and train_rl_service.py, which has no such argument, train none)."""
    rank = getattr(args, 'vit_lora_rank', ADDED_SETTINGS['vit_lora_rank'])
    if not rank:
        return 0, None, (), None
    return rank, args.vit_lora_alpha, tuple(args.vit_lora_targets.split(',')), args.vit_lora_sdpa


VIT_SDPA_CALLS = {'efficient': 0, 'math': 0}  # ViT attention calls through a pinned kernel (use_vit_sdpa)


def use_vit_sdpa(visual, choice):
    """--vit-lora-sdpa: which SDPA kernel the ViT runs with a vision LoRA (measured on B200, check_vit_lora.py and
    the 16 Oct probes in the vision-LoRA notes).

    ``'auto'`` (default): torch's pick, cuDNN -- exactly the frozen-ViT trainer's forward (a B = 0 vision LoRA leaves
    loss and text-LoRA gradients bitwise unchanged), fastest; its backward is not run-to-run deterministic (vision-LoRA
    gradients differ by ~1 % rel-L2 between identical runs), so a resume or repeat is not bitwise.
    ``'efficient'``: the memory-efficient kernel; its backward was bitwise reproducible for images up to ~1k patches
    (a 30x30 ARC grid at 16 px per cell is 900) but not reliably at 4k and never at 9k patches, deterministic-algorithms
    mode or not; its forward differs from cuDNN by bf16 rounding, which the ViT amplifies to ~5 % rel-L2 at its output
    (the loss of a short bench step moved by 1.4 %).
    ``'math'``: deterministic at every size; materializes the attention matrix (a 9k-patch image: ~15 GB per block in
    backward with the block checkpointing) and is 2.5-5x slower in the ViT.
    The deployed SGLang ViT runs FA4, none of these. The kernel is pinned per call, inside the attention function, so the
    block checkpoint recompute in backward runs the same kernel as the forward."""
    if choice == 'auto':
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    backend = {'efficient': SDPBackend.EFFICIENT_ATTENTION, 'math': SDPBackend.MATH}[choice]

    def pinned(*args, **kwargs):
        VIT_SDPA_CALLS[choice] += 1
        with sdpa_kernel([backend]):
            return sdpa_attention_forward(*args, **kwargs)

    name = f'sdpa_vit_{choice}'
    AttentionInterface.register(name, pinned)
    configs = {id(block.attn.config) for block in visual.blocks} | {id(visual.config)}
    if len(configs) != 1 or visual.config._attn_implementation != 'sdpa':
        raise RuntimeError(f'ViT attention config: {len(configs)} objects, {visual.config._attn_implementation!r}')
    visual.config._attn_implementation = name


def check_routing_spec(spec, replay, text_config, where):
    """The data's routing (manifest ``routing``) must have this model's layers, top-k and expert set."""
    if (int(spec['layers']), int(spec['top_k'])) != (replay.layers, replay.top_k):
        raise RuntimeError(f'{where}: routing {(spec["layers"], spec["top_k"])}, model ({replay.layers}, {replay.top_k})')
    # expert ids are only meaningful in the rollout model's expert set (a REAP model renumbers a subset)
    if spec['num_experts'] != text_config.num_experts:
        raise RuntimeError(f'{where}: routing recorded on a {spec["num_experts"]}-expert model, '
                           f'this model has {text_config.num_experts}')


def assert_step_kernels(batch, audit, gdn_calls, update, offload, frozen=None):
    """After a forward (and backward): every QSA layer ran the sparse selection on audited runs, every GDN call was
    FlashQLA (36 layers, plus the checkpoint recompute when training), the residual offload executed, every dense-QAT
    projection ran its QAT forward (fp8_dense_qat.assert_step_calls; nothing to check without --dense-qat), every
    --fast-kernels module ran its kernel (trainer_kernels.assert_step_calls). ``frozen`` (the full-weight RL trainer,
    dense_full_rl.FullWeights.frozen): {'layers', 'gdn', 'dense_fp8'} of the first decoder layers no gradient reaches,
    which run one forward and no recompute or backward; their full-weight QAT projections are checked here too."""
    frozen = frozen or {'layers': 0, 'gdn': 0, 'dense_fp8': None, 'stream_grad': True}
    fp8_dense_qat.assert_step_calls(update)
    if frozen['dense_fp8'] is not None:
        import dense_full_qat
        dense_full_qat.assert_step_calls(update, frozen=frozen['dense_fp8'])
    trainer_kernels.assert_step_calls(update, frozen['layers'], frozen['gdn'], frozen['stream_grad'])
    from int3_moe import assert_step_calls as int3_step_calls  # lazily: it loads Triton
    int3_step_calls(update, frozen['layers'])
    if audit and len(batch['packed_meta']['qsa_audit']) != 12 and batch['packed_meta']['max_length'] > 2048:
        raise RuntimeError('not every QSA layer ran the sparse selection')
    if gdn_calls != ((36 - frozen['gdn']) * 2 + frozen['gdn'] if update else 36):
        raise RuntimeError(f'{gdn_calls} FlashQLA calls this step; another GDN backend ran')
    if offload is not None and offload.tensors < 48 - frozen['layers']:
        raise RuntimeError(f'checkpoint residual offload did not execute: {offload.tensors}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=['train', 'bench', 'score'], required=True)
    parser.add_argument('--score-keys', help='JSON list of dataset keys (either split) to score, one segment per forward')
    parser.add_argument('--score-variants', default='deployed',
                        help='comma list of deployed | semantic_ple | hf_vit (controls); one scores/<variant>/ each')
    parser.add_argument('--bench-steps', help='JSON list of steps, each a list of world bins of train indices')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--steps', type=int, default=0, help='train: stop after this many updates (0 = one epoch)')
    parser.add_argument('--bin-tokens', type=int, default=120_000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    add_lora_target_arguments(parser)
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--base-storage', choices=['bf16', 'fp8', 'int3'], default='fp8',
                        help='int3 = gEAN320 INT3 package with BF16 dense weights (W3A8 experts, dX on dequantized BF16)')
    parser.add_argument('--offload-checkpoint-inputs', action='store_true')
    parser.add_argument('--offload-mode', choices=['sync', 'async'], default='async',
                        help='async: side-stream D2H, backward prefetch, forward saves only (checkpoint_offload.py)')
    parser.add_argument('--offload-keep-gib', type=float, default=0.0,
                        help='async: the newest checkpointed layer inputs stay on the GPU while they fit in this many GiB; only older ones go to host memory (checkpoint_offload.py keep_bytes)')
    parser.add_argument('--fp8-dx', choices=['fp8', 'bf16'], default='fp8',
                        help='FP8 base: backward GEMM precision (fp8_moe.set_dx_mode)')
    parser.add_argument('--int3-act', choices=['fp8', 'bf16'], default=ADDED_SETTINGS['int3_act'],
                        help='INT3 base: activation format of the routed experts\' forward (int3_moe.py), as '
                        'train_rl_service.py: fp8 = W3A8, bf16 = W3A16 like Humming A16 (ARC_INT3_ACT=bf16)')
    parser.add_argument('--ple-media', choices=['deployed', 'semantic'], default='deployed')
    parser.add_argument('--ple-codec', choices=['fp8', 'int5'], default='fp8',
                        help='PLE table storage: fp8 = the checkpoint rows x weight_scale; int5 = re-quantized at load to the INT5 table '
                        'the deployed server builds with ARC_PLE_INT5=1 (lookups bit-identical to it); needs --base-storage fp8 or int3')
    parser.add_argument('--vit-interp', choices=['deployed', 'hf'], default='deployed')
    parser.add_argument('--qsa-backend', default='auto',
                        help="'auto' = gather kernel; 'flex' = best Flex mask; or an explicit sparse_qsa backend")
    parser.add_argument('--audit-steps', type=int, default=2, help='QSA coverage audit on the first N runs')
    parser.add_argument('--dump-grads', action='store_true', help='rank 0 saves the synced LoRA gradients of run 1')
    parser.add_argument('--decisions', choices=['record', 'replay'],
                        help='record or force MoE/QSA discrete decisions (decision_replay.py) for kernel A/B')
    parser.add_argument('--decisions-dir')
    parser.add_argument('--verify-offload', action='store_true', help='debug: checksum sync offload pack vs unpack')
    parser.add_argument('--ple-crosscheck', action='store_true',
                        help='debug: on audited runs also run the padded PLE path and record the difference')
    parser.add_argument('--recompute-probe-layer', type=int, help='debug: raise at the first submodule of this '
                        'layer whose checkpoint recompute differs from its forward (probes/recompute_probe.py)')
    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--eval-steps', type=int, default=4)
    parser.add_argument('--save-every', type=int, default=0)
    parser.add_argument('--lr-warmup-steps', type=int, default=20)
    parser.add_argument('--min-lr-ratio', type=float, default=0.1, help='cosine floor as a fraction of --lr')
    parser.add_argument('--resume', help='train: a checkpoint-<step> directory written by --save-every')
    parser.add_argument('--routing-replay', choices=['none', 'observe', 'force'], default='none',
                        help='MoE routing from the rollout (routing_replay.py; data built with prepare_tito_mm.py '
                             '--routing): force = take it, observe = route naturally and count agreement')
    parser.add_argument('--score-temperature', type=float, default=1.0,
                        help='score: log-probabilities of logits/T (TITO ledger logprobs are after the sampling T)')
    parser.add_argument('--init-lora', help='start from the LoRA of a checkpoint-<step> directory (trainer_state.pt)')
    parser.add_argument('--save-merged-every', type=int, default=0,
                        help='train: publish the LoRA-merged adapted projections every N steps and at the last '
                             '(merged_export.py) to <out>/merged/v<step>')
    parser.add_argument('--vit-lora-rank', type=int, default=ADDED_SETTINGS['vit_lora_rank'],
                        help='>0: also train a LoRA of this rank on the ViT (base ViT weights stay frozen; the ViT runs '
                             'with autograd and block checkpointing); 0 = frozen ViT')
    parser.add_argument('--vit-lora-alpha', type=int, default=ADDED_SETTINGS['vit_lora_alpha'])
    parser.add_argument('--vit-lora-targets', default=ADDED_SETTINGS['vit_lora_targets'],
                        help=f'comma list of {"/".join(VIT_TARGETS)}: blocks = attn.qkv/proj + mlp.linear_fc1/fc2 of '
                             'every ViT block, merger = the patch merger\'s linear_fc1/fc2')
    parser.add_argument('--vit-lr', type=float, default=ADDED_SETTINGS['vit_lr'],
                        help='peak LR of the vision LoRA (same schedule; default --lr)')
    parser.add_argument('--vit-lora-sdpa', choices=['auto', 'efficient', 'math'], default=ADDED_SETTINGS['vit_lora_sdpa'],
                        help="ViT attention kernel with a vision LoRA (use_vit_sdpa): auto = torch's pick (cuDNN, the "
                             'frozen-ViT forward; nondeterministic backward); efficient = reproducible for small images '
                             '(<= ~1k patches); math = always reproducible, memory-heavy')
    parser.add_argument('--vit-aux-color-loss', type=float, default=ADDED_SETTINGS['vit_aux_color_loss'],
                        help='>0: add this weight x the mean CE of a trained Linear head reading every merger token\'s '
                             'four 16x16 quadrant colours (ARC 0-9 or white padding, from the pixels; 16pad-rendered '
                             'data only; needs --vit-lora-rank; the head trains in the vision LoRA\'s param group and '
                             'is not exported)')
    fp8_dense_qat.add_arguments(parser, debug=True)
    trainer_kernels.add_arguments(parser, debug=True)
    args = parser.parse_args()
    if args.vit_lora_rank < 0 or (not args.vit_lora_rank and (
            (args.vit_lora_alpha, args.vit_lora_targets, args.vit_lr, args.vit_lora_sdpa) != tuple(
                ADDED_SETTINGS[k] for k in ('vit_lora_alpha', 'vit_lora_targets', 'vit_lr', 'vit_lora_sdpa')))):
        parser.error('--vit-lora-alpha/--vit-lora-targets/--vit-lr/--vit-lora-sdpa need --vit-lora-rank > 0')
    if args.vit_lr is not None and args.vit_lr < 0:
        parser.error('invalid --vit-lr')
    if args.vit_aux_color_loss < 0 or (args.vit_aux_color_loss and (not args.vit_lora_rank or args.mode == 'score')):
        parser.error('--vit-aux-color-loss is a non-negative train/bench weight and needs --vit-lora-rank > 0')
    fp8_dense_qat.check_arguments(parser, args, debug=True)
    trainer_kernels.check_arguments(parser, args, debug=True)
    if args.fast_kernels_shadow and args.mode != 'bench':
        parser.error('--fast-kernels-shadow is a bench-mode check')
    if args.dense_qat_check_export and (args.mode != 'train' or not args.save_merged_every):
        parser.error('--dense-qat-check-export is a train-mode check of --save-merged-every exports')
    if args.mode == 'bench' and (not args.bench_steps or args.warmup < 1 or args.repeats < 2):
        parser.error('bench needs --bench-steps, >=1 warmup and >=2 repeats')
    if bool(args.decisions) != bool(args.decisions_dir) or (args.decisions and args.mode != 'bench'):
        parser.error('--decisions needs --decisions-dir and bench mode')
    if (args.mode == 'score') != bool(args.score_keys):
        parser.error('--score-keys goes with --mode score')
    if args.lr < 0 or args.steps < 0 or args.bin_tokens < 4096:
        parser.error('invalid LR/steps/bin size')
    if args.resume and args.mode != 'train':
        parser.error('--resume is for train mode')
    if args.lr_warmup_steps < 0 or not 0 <= args.min_lr_ratio <= 1:
        parser.error('invalid LR schedule')
    if args.routing_replay != 'none' and args.decisions:
        parser.error('--routing-replay and --decisions both patch the MoE gates')
    if args.init_lora and args.resume:
        parser.error('--init-lora starts a run; --resume continues one')
    if args.int3_act != 'fp8' and args.base_storage != 'int3':
        parser.error('--int3-act bf16 needs --base-storage int3')
    if args.save_merged_every < 0 or (args.save_merged_every and args.mode != 'train'):
        parser.error('--save-merged-every is a train-mode step interval')
    if not args.score_temperature > 0 or (args.score_temperature != 1.0 and args.mode != 'score'):
        parser.error('--score-temperature is a positive score-mode setting')

    from checkpoint_offload import CheckpointInputOffload
    from distributed_utils import assert_replicated, sync_gradients
    from tito_bins import cyclic_pad, plan_eval_steps, plan_steps, step_balance
    from tito_mm_data import TitoMMDataset

    from arcforge.train.preflight import assert_flash_qla_stack, assert_gdn_backend, assert_liger_moe_patch

    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    world = dist.get_world_size()
    # 3 ranks since 10-15 (5 rollout + 3 trainer on one B200 box): FSDP2 pads the uneven dim-0 shards (320 experts ->
    # 107 each), the PLE/embedding row shards split total*rank//world.
    if world not in (2, 3, 4):
        raise RuntimeError('FSDP over 2, 3 or 4 ranks only')
    assert_flash_qla_stack()
    assert_gdn_backend('flash_qla')  # the fla Triton fallback is not run-to-run deterministic on B200
    assert_liger_moe_patch()
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'code').mkdir()
        for path in Path(__file__).parent.glob('*.py'):
            shutil.copy2(path, out / 'code' / path.name)
        (out / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    dist.barrier()
    with_routing = args.routing_replay != 'none'
    train_data = TitoMMDataset(args.data, 'train', routing=with_routing)
    eval_data = TitoMMDataset(args.data, 'validation', routing=with_routing)
    tokenizer_sha = hashlib.sha256((Path(args.model) / 'tokenizer.json').read_bytes()).hexdigest()
    if train_data.tokenizer_sha256 != tokenizer_sha:
        raise RuntimeError('dataset tokenizer does not match the checkpoint')
    media_device = train_data.manifest['processor'].get('device')  # v1 (CPU-emulated) manifests predate the field
    if args.mode == 'train' and media_device != 'cuda':
        raise RuntimeError(f'{args.data}: media were not processed as deployed (processor device {media_device!r}); '
                           'the CPU emulation differs by +-1 u8 -- regenerate with DEVICE=cuda prepare_tito_mm.sh')
    aux_weight = args.vit_aux_color_loss
    if aux_weight:  # labels come from the pixels of the 16pad grid renderer; any other image would be mislabelled
        for data in (train_data, eval_data):
            renderer = data.manifest['processor'].get('renderer')  # absent: not rendered by the grid renderer
            if renderer != '16pad':
                raise RuntimeError(f'--vit-aux-color-loss needs 16pad-rendered grid images; {args.data} {data.split}: '
                                   f'renderer {renderer!r}')
    trainer = load_trainer(args, rank, device, world, args.routing_replay, bool(args.save_merged_every),
                           shadow_kernels=args.fast_kernels_shadow)
    model, base, parameters, names = trainer.model, trainer.base, trainer.parameters, trainer.names
    loading, kernels, gdn, replay = trainer.loading, trainer.kernels, trainer.gdn, trainer.replay
    exporter, text_config, vision_config = trainer.exporter, trainer.text_config, trainer.vision_config
    width, slots = trainer.width, trainer.slots
    if replay is not None:
        for data in (train_data, eval_data):
            check_routing_spec(data.routing_spec, replay, text_config, args.data)
    if args.verify_offload:
        import checkpoint_offload
        checkpoint_offload.VERIFY['on'] = True
    if args.recompute_probe_layer is not None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent / 'probes'))
        from recompute_probe import install as install_recompute_probe
        install_recompute_probe(base.model.language_model.layers[args.recompute_probe_layer],
                                f'layer {args.recompute_probe_layer} rank {rank}')
    decisions = None
    if args.decisions:
        from decision_replay import Decisions
        decisions = Decisions(args.decisions, args.decisions_dir, rank)
        decisions.install(base.model.language_model)
    vision_parameters = trainer.vision_parameters  # the vision LoRA, then the colour head if any
    aux_head = trainer.aux_head
    peak = args.lr if args.mode == 'train' else 0.0
    if vision_parameters:  # its own param group (one LR schedule factor for both)
        vit_peak = (args.lr if args.vit_lr is None else args.vit_lr) if args.mode == 'train' else 0.0
        text_parameters = parameters[:len(parameters) - len(vision_parameters)]
        optimizer = torch.optim.AdamW([{'params': text_parameters}, {'params': vision_parameters, 'lr': vit_peak}],
                                      lr=peak, weight_decay=0, fused=True)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=peak, weight_decay=0, fused=True)
    print(f'[rank{rank}] ready: {sum(p.numel() for p in parameters)} trainable', flush=True)

    runs = [0]

    def run(step_bins, data, update):
        rows = [data[i] for i in step_bins[rank]]
        model.train(update)
        # The ViT has no dropout; its train mode only turns on the per-block activation checkpointing
        # (gradient_checkpointing_enable), which a ViT with autograd needs at real media counts.
        base.model.visual.train(update and bool(vision_parameters))
        model.zero_grad(set_to_none=True)
        if aux_head is not None:  # not a submodule of model: without this its gradient accumulates across steps
            aux_head.zero_grad(set_to_none=True)
        audit = runs[0] < args.audit_steps
        runs[0] += 1
        batch, media = build_batch(rows, 0, device, vision_config, text_config.vocab_size, ple_media=args.ple_media,
                                   color_labels=bool(aux_weight),
                                   vit_interp=args.vit_interp, audit=audit)
        batch['packed_meta']['qsa_mask_backend'] = qsa_backend(args.qsa_backend, rows)
        if args.ple_crosscheck and audit:
            from packed_qwen4 import RuntimeRecords
            batch['packed_meta']['ple_crosscheck'] = RuntimeRecords()
        local_tokens = batch['input_ids'].shape[1]
        local_supervised = int((batch['labels'][:, 1:] != IGNORE).sum())
        counts = torch.tensor([local_supervised, local_tokens], device=device, dtype=torch.float64)
        if aux_weight:  # third entry: colour-labelled quadrants of all ranks (the aux loss is their global mean)
            quadrants = 0 if media is None else media['color_labels'].numel()
            counts = torch.cat([counts, counts.new_tensor([quadrants])])
        dist.all_reduce(counts)
        if counts[0] == 0:
            raise RuntimeError('no supervised tokens in this step')
        if replay is not None:
            replay.begin(torch.from_numpy(np.concatenate([row['routing'] for row in rows])).to(device))
        offload = None
        if args.offload_checkpoint_inputs and update:
            offload = CheckpointInputOffload(local_tokens, width, slots, asynchronous=args.offload_mode == 'async',
                                             keep_bytes=int(args.offload_keep_gib * 2**30))
        prior = dict(kernels)
        gdn_before = gdn['calls']
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # allocated and reserved peaks, per step
        started = time.perf_counter()
        if decisions is not None:
            decisions.begin([row['key'] for row in rows])
        context = torch.enable_grad() if update else torch.no_grad()
        with context, (nullcontext() if offload is None else offload.context()):
            if aux_weight:
                output = model(**batch, media=media, aux_color_head=aux_head)
                aux = {k: v for k, v in output.items() if k.startswith('aux_')}
            else:
                output = model(**batch, media=media)
            loss_sum = output['loss'] if aux_weight else output.loss
            if not torch.isfinite(loss_sum) or (aux_weight and not torch.isfinite(aux['aux_sum'])):
                raise RuntimeError('nonfinite loss')
            if update:
                objective = loss_sum / counts[0]
                if aux_weight and counts[2] > 0:
                    objective = objective + aux_weight * aux['aux_sum'] / counts[2]
                objective.backward()
        torch.cuda.synchronize()
        compute = time.perf_counter() - started
        if decisions is not None:
            decisions.finish()
        routing = None if replay is None else {k: v.tolist() for k, v in replay.finish().items() if k != 'calls'}
        norm = vision_norm = None
        if update:
            if vision_parameters and media is None:  # this rank's loss does not depend on the ViT: its gradient is 0
                for p in vision_parameters:
                    if p.grad is not None:
                        raise RuntimeError('vision LoRA gradient without media')
                    p.grad = torch.zeros_like(p)
            for name, p in zip(names, parameters, strict=True):
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise RuntimeError(f'missing/nonfinite gradient: {name}')
            sync_gradients(parameters)
            if args.dump_grads and runs[0] == 1 and rank == 0:
                from safetensors.torch import save_file
                save_file({n: p.grad.detach().float().cpu().contiguous() for n, p in zip(names, parameters, strict=True)},
                          str(out / 'grads-run1.safetensors'))
            if vision_parameters:  # before clipping, like gradient_norm
                vision_norm = float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in vision_parameters])))
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            if not norm > 0:
                raise RuntimeError('zero gradient')
            optimizer.step()
            assert_replicated(parameters)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        assert_step_kernels(batch, audit, gdn['calls'] - gdn_before, update, offload)
        local = {'rank': rank, 'keys': [row['key'] for row in rows], 'tokens': local_tokens,
                 **({k: float(v.detach()) for k, v in aux.items()} if aux_weight else {}),
                 'supervised': local_supervised, 'media': 0 if media is None else int(media['grid_thw'].shape[0]),
                 'loss_sum': float(loss_sum.detach()), 'compute_seconds': compute, 'seconds': seconds,
                 'peak_gib': torch.cuda.max_memory_allocated() / 2**30,
                 'reserved_gib': torch.cuda.max_memory_reserved() / 2**30, 'host_rss_gib': host_rss_gib(),
                 'offloaded_gib': 0.0 if offload is None else offload.bytes / 2**30,
                 'pinned_gib': 0.0 if slots is None else slots.bytes / 2**30,
                 'pinned_slots': {} if slots is None else slots.counts,
                 'kernels': {k: kernels[k] - prior[k] for k in kernels if isinstance(kernels[k], int)},
                 'decisions': None if decisions is None else dict(decisions.counts),
                 'fast_kernels_shadow': {k: dict(v) for k, v in trainer_kernels.SHADOW.items()},
                 'routing': routing,
                 'ple_crosscheck': None if batch['packed_meta']['ple_crosscheck'] is None
                 else list(batch['packed_meta']['ple_crosscheck'].values())}
        ranks = [None] * world
        dist.all_gather_object(ranks, local)
        wall = max(r['seconds'] for r in ranks)
        agreement = {}
        if replay is not None:  # rows whose natural top-k equals the rollout's, and mean shared experts (of top_k)
            rows_compared = sum(sum(r['routing']['rows']) for r in ranks)
            agreement = {'routing_equal_rate': sum(sum(r['routing']['equal']) for r in ranks) / rows_compared,
                         'routing_mean_shared': sum(sum(r['routing']['overlap']) for r in ranks) / rows_compared}
        return {'loss': sum(r['loss_sum'] for r in ranks) / float(counts[0]), 'seconds': wall, **agreement,
                **(aux_summary(ranks) if aux_weight else {}),
                'real_tok_s': float(counts[1]) / wall, 'supervised_tok_s': float(counts[0]) / wall,
                'compute_imbalance': max(r['compute_seconds'] for r in ranks) / statistics.mean(
                    r['compute_seconds'] for r in ranks),
                'gradient_norm': norm, **({'vision_gradient_norm': vision_norm} if vision_parameters else {}),
                'ranks': ranks}

    logs = []

    def record(entry):
        logs.append(entry)
        if rank == 0:
            print(json.dumps({k: v for k, v in entry.items() if k != 'ranks'}), flush=True)
            (out / 'progress.json').write_text(json.dumps(logs, indent=1) + '\n')

    if args.mode == 'score':
        variants = {'deployed': ('deployed', 'deployed'), 'semantic_ple': ('semantic', 'deployed'),
                    'hf_vit': ('deployed', 'hf')}
        chosen_variants = args.score_variants.split(',')
        if not set(chosen_variants) <= set(variants):
            raise ValueError(f'unknown score variants {chosen_variants}')
        wanted = json.loads(Path(args.score_keys).read_text())
        where = {key: (data, i) for data in (train_data, eval_data) for i, key in enumerate(data.keys)}
        missing = [k for k in wanted if k not in where]
        if missing:
            raise KeyError(f'{len(missing)} score keys not in the dataset: {missing[:3]}')
        if len(set(wanted)) != len(wanted):
            raise ValueError('duplicate score keys: two ranks would write the same file')
        chosen, own = cyclic_pad(wanted, world)  # every rank forwards each step; padding is not written
        model.eval()
        for variant in chosen_variants:
            ple_media, vit_interp = variants[variant]
            scores = out / 'scores' / variant
            if rank == 0:
                scores.mkdir(parents=True)
            dist.barrier()
            for start in range(0, len(chosen), world):
                position = start + rank
                data, i = where[chosen[position]]
                row = data[i]
                batch, media = build_batch([row], 0, device, vision_config, text_config.vocab_size, score=True,
                                           ple_media=ple_media, vit_interp=vit_interp)
                batch['packed_meta']['qsa_mask_backend'] = qsa_backend(args.qsa_backend, [row])
                if replay is not None:
                    replay.begin(torch.from_numpy(row['routing']).to(device))
                with torch.no_grad():
                    result = model(**batch, media=media, score=True, temperature=args.score_temperature)
                routing = None if replay is None else {k: v.tolist() for k, v in replay.finish().items() if k != 'calls'}
                if not bool(result['valid'].all()) or len(result['token_logprobs']) != len(row['input_ids']) - 1:
                    raise RuntimeError('score must cover every next-token position of a single segment')
                if own[position]:
                    name = row['key'].replace('/', '_').replace(':', '__')
                    np.save(scores / f'{name}.npy', result['token_logprobs'].cpu().numpy())
                    if routing is not None:
                        (scores / f'{name}.routing.json').write_text(json.dumps(routing) + '\n')
                    record({'phase': 'score', 'variant': variant, 'key': row['key'], 'rank': rank,
                            'tokens': len(row['input_ids']), 'mean_logprob': float(result['token_logprobs'].mean())})
                dist.barrier()
    elif args.mode == 'bench':
        steps = json.loads(Path(args.bench_steps).read_text())
        if any(len(step) != world for step in steps):
            raise ValueError('every bench step needs one bin per rank')
        for index, step in enumerate(steps):
            for repeat in range(args.warmup + args.repeats):
                record({'phase': 'warmup' if repeat < args.warmup else 'measure', 'step': index,
                        **run(step, train_data, True)})
    else:
        import math
        steps, dropped = plan_steps(train_data.lengths, args.bin_tokens, world, args.seed)
        eval_steps = plan_eval_steps(eval_data.lengths, args.bin_tokens, world, args.seed)[:args.eval_steps]
        if args.eval_every and not eval_steps:
            raise RuntimeError('--eval-every needs a non-empty validation split')
        plan_sha = hashlib.sha256(json.dumps([steps, train_data.keys]).encode()).hexdigest()
        if rank == 0:
            (out / 'plan.json').write_text(json.dumps({'steps': len(steps), 'dropped_bins': dropped, 'sha256': plan_sha,
                'balance_p90': float(np.percentile([step_balance(s, train_data.lengths) for s in steps], 90))}) + '\n')
        limit = args.steps or len(steps)
        if limit > len(steps):
            raise ValueError(f'--steps {limit} exceeds the {len(steps)}-step epoch plan')
        warm = min(args.lr_warmup_steps, limit)

        def lr_factor(done):  # `done` optimizer steps taken so far
            if done < warm:
                return (done + 1) / warm
            progress = (done - warm) / max(1, limit - warm)
            return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        first = 1
        if args.resume:
            state = torch.load(Path(args.resume) / 'trainer_state.pt', map_location='cpu', weights_only=False)
            if state['plan_sha256'] != plan_sha or state['limit'] != limit:
                raise RuntimeError('resume needs the same data, seed, bin size and --steps as the saved run')
            conflicts = resume_conflicts(state['args'], vars(args))
            if conflicts:
                raise RuntimeError(f'resume would change training settings (saved, now): {conflicts}')
            if set(state['lora']) != set(names):
                raise RuntimeError('saved LoRA parameter names differ from this model')
            with torch.no_grad():
                for name, p in zip(names, parameters, strict=True):
                    p.copy_(state['lora'][name].to(p.device))
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            first = state['step'] + 1
            assert_replicated(parameters)
            if rank == 0:
                print(f'resumed {args.resume} at step {state["step"]}', flush=True)

        def evaluate(iteration):
            loss_sum = supervised = 0
            routing = {'rows': 0, 'equal': 0, 'overlap': 0}
            scored_ranks = []
            for step, scored in eval_steps:
                for r in run(step, eval_data, False)['ranks']:
                    if scored[r['rank']]:  # a padded rank repeats a bin that another rank scores
                        scored_ranks.append(r)
                        loss_sum += r['loss_sum']
                        supervised += r['supervised']
                        if r['routing'] is not None:
                            routing = {k: routing[k] + sum(r['routing'][k]) for k in routing}
            agreement = {} if not routing['rows'] else {'routing_equal_rate': routing['equal'] / routing['rows'],
                                                        'routing_mean_shared': routing['overlap'] / routing['rows']}
            record({'phase': 'eval', 'step': iteration, 'loss': loss_sum / supervised, **agreement,
                    **(aux_summary(scored_ranks) if aux_weight else {})})

        def save(iteration):
            dist.barrier()
            if rank == 0:
                target = out / f'checkpoint-{iteration}'
                state = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
                model.save_pretrained(target, state_dict=state, safe_serialization=True, save_embedding_layers=False)
                torch.save({'step': iteration, 'limit': limit, 'plan_sha256': plan_sha,
                            'lora': {n: p.detach().cpu() for n, p in zip(names, parameters, strict=True)},
                            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                            'args': vars(args)}, target / 'trainer_state.pt')
            dist.barrier()

        def save_merged(iteration):
            dist.barrier()
            if rank == 0:
                from merged_export import adapter_tensors, changed_elements, lora_pairs
                started = time.perf_counter()
                pairs = lora_pairs(adapter_tensors({n: p.detach().cpu() for n, p in zip(names, parameters, strict=True)}))
                manifest = exporter.write(out / 'merged' / f'v{iteration:06d}', pairs, iteration, {'run': str(out)})
                record({'phase': 'merged', 'step': iteration, 'weight_version': manifest['weight_version'],
                        'bytes': manifest['bytes'], 'seconds': time.perf_counter() - started,
                        'changed_bf16_elements': changed_elements(manifest)})  # stored elements (v2: e4m3 + scales too)
                fp8_dense_qat.CAPTURE = None
            dist.barrier()

        if args.eval_every and first == 1:
            evaluate(0)
        for iteration in range(first, limit + 1):
            lr = optimizer.param_groups[0]['lr']
            record({'phase': 'train', 'step': iteration, 'lr': lr,
                    **({'vit_lr': optimizer.param_groups[1]['lr']} if vision_parameters else {}),
                    'balance': step_balance(steps[iteration - 1], train_data.lengths),
                    **run(steps[iteration - 1], train_data, True)})
            scheduler.step()
            if args.eval_every and (iteration % args.eval_every == 0 or iteration == limit):
                evaluate(iteration)
            if args.save_every and (iteration % args.save_every == 0 or iteration == limit):
                save(iteration)
            if args.save_merged_every and (iteration % args.save_merged_every == 0 or iteration == limit):
                if args.dense_qat_check_export:  # one more forward of the LoRA exported next: the export must match it
                    if rank == 0:
                        fp8_dense_qat.CAPTURE = {}
                    run(steps[iteration - 1], train_data, False)
                save_merged(iteration)
    if not gdn['calls'] or not (kernels['sparse_flex_calls'] + kernels['sparse_block_calls'] + kernels['sparse_gather_calls']
                                + kernels['fa4_calls']):
        raise RuntimeError('GDN or QSA kernels never executed')
    if rank == 0:
        from fp8_moe import COUNTERS
        from int3_moe import COUNTERS as INT3_COUNTERS
        routing_summary = None
        forwards = [r['routing'] for e in logs for r in e.get('ranks', ()) if r.get('routing')]
        if forwards:
            from routing_replay import summarize
            routing_summary = {'mode': args.routing_replay, **summarize(
                [{k: torch.tensor(v) for k, v in f.items()} for f in forwards], replay.top_k)}
        (out / 'result.json').write_text(json.dumps({
            'status': 'PASS', 'mode': args.mode, 'loading': loading, 'kernels': kernels, 'gdn_calls': gdn['calls'],
            'fp8_moe_counters': dict(COUNTERS), 'int3_moe_counters': dict(INT3_COUNTERS),
            'trainable_names': len(names), 'routing': routing_summary,
            **({'vision_trainable_names': len(trainer.vision_names)} if vision_parameters else {}),
            'records': logs}, indent=1) + '\n')
    if slots is not None:
        slots.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
