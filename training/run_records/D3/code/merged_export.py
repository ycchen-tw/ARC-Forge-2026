"""LoRA-merged weights of the adapted projections, as a directory SGLang's ``update_weights_from_disk`` can load.

Only the adapted tensors are written (228 for train_tito_mm.py: 12 QSA layers x q/k/v/o, 36 GDN layers x
in_proj_qkv/z/a/b and out_proj; ~5 GiB BF16; 372 with ``--lora-shared-expert``: + 48 layers x mlp.shared_expert.gate/up/
down_proj, +0.44 GiB BF16), one ``model.safetensors`` under the checkpoint's own keys
(``model.language_model.layers.N...weight``); with ``--vit-lora-rank`` also the adapted ViT weights
(``model.visual.blocks.N.attn.qkv.weight`` ..., ``model.visual.merger.linear_fc1.weight``; BF16 in every FN checkpoint,
merged with the vision LoRA's own rank / alpha, recorded as the manifest's ``vision_lora``; biases are not adapted).
SGLang's loader globs ``*.safetensors`` in the directory and its Qwen4 ``load_weights`` takes a subset of keys (stacked
q/k/v, in_proj and the shared expert's gate/up shards of gate_up_proj load independently); MTP keys are never written.
The attention projections are BF16 in every FN checkpoint (FP8 and NVFP4 alike), the shared expert in the gEAN320 INT3
package, so no requantization (the base must be 2-D BF16).

v2 (``dense_fp8_keys`` given, the trainer's ``--dense-qat fp8``): the projections the deployed model runs as block FP8 are
written as ``<key>`` (float8_e4m3fn) plus ``<key minus .weight>.weight_scale_inv`` (fp32 [N/128, K/128]), the deployed
format, computed by fp8_dense_qat's ``quantize_block_fp8(effective_weight(...))`` on the GPU -- the functions and device the
trainer's forward quantizes with, so these are the tensors its last forward multiplied by. The other projections stay BF16.

Merge: ``(W.float() + alpha/r * (B.float() @ A.float())).to(bf16)``, always from the original base read from disk,
so rounding does not compound across syncs. It differs from the trainer's LoRA forward (bf16 base output plus the
fp32 LoRA path) by the one bf16 rounding of the sum; the manifest counts the elements the delta changed.

Full mode (``full=True``, train_rl_service ``--trainable full``, dense_full_rl.py): no LoRA; ``write_full`` takes the
trainer's BF16 weights of the trained keys (the cast of its FP32 masters the forward multiplies) and writes every key
of the set the same way -- block FP8 for ``dense_fp8_keys`` (quantize_block_fp8 on the GPU, as dense_full_qat's
forward), BF16 for the rest -- the keys it was not given (frozen layers) from the base. Same v2 layout and format; the
manifest carries ``trainer: full`` and ``lora_rank`` / ``lora_alpha`` / ``scale`` null.

A version is published as ``<dir>.tmp`` (written, read back bitwise, manifest), renamed to ``<dir>``, then ``DONE``
is written: a consumer loads a directory only once ``DONE`` exists. The manifest carries ``weight_version`` (the
optimizer step) and per-tensor sha256 of the base and the merged bytes.
"""

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from multimodal import adapter_checkpoint_key
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FORMAT = 'fn-lora-merged-v1'
FORMAT_V2 = 'fn-lora-merged-v2'
FP8_FORMAT = 'fp8_e4m3_block128'
SCALE_FORMAT = 'fp32_scale_inv'
ADAPTER = 'default'  # PEFT adapter name of attach_replicated_lora
VISION_KEYS = ('model.visual.blocks.', 'model.visual.merger.')  # train_tito_mm --vit-lora-rank projections


# Trainer-only parameters that are saved with the LoRA but never exported (train_tito_mm --vit-aux-color-loss head)
TRAINER_ONLY = ('vit_aux_color_head.',)


def adapter_tensors(named):
    """``named`` without the trainer-only parameters (TRAINER_ONLY): what lora_pairs exports."""
    return {k: v for k, v in named.items() if not k.startswith(TRAINER_ONLY)}


def lora_pairs(named):
    """{checkpoint key: (A, B)} from PEFT parameter names (``...<module>.lora_A.default.weight``)."""
    modules = defaultdict(dict)
    for name, tensor in named.items():
        for part in ('lora_A', 'lora_B'):
            suffix = f'.{part}.{ADAPTER}.weight'
            if name.endswith(suffix):
                modules[name[:-len(suffix)]][part] = tensor
                break
        else:
            raise ValueError(f'not a LoRA weight of adapter {ADAPTER!r}: {name}')
    pairs = {}
    for module, parts in modules.items():
        if set(parts) != {'lora_A', 'lora_B'}:
            raise ValueError(f'{module}: has {sorted(parts)}')
        key = adapter_checkpoint_key(module + '.lora_A.weight')
        if not key.startswith(VISION_KEYS) and (
                '.mtp.' in key or key.startswith('mtp.') or not key.startswith('model.language_model.layers.')):
            raise ValueError(f'{module} -> {key}: not a decoder-layer or ViT projection')
        pairs[key] = (parts['lora_A'], parts['lora_B'])
    return pairs


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy()).hexdigest()


def scale_key(key):
    """Checkpoint key of the block scale of an FP8 weight key: ``<module>.weight`` -> ``<module>.weight_scale_inv``."""
    return key.removesuffix('.weight') + '.weight_scale_inv'


def same_tensor(a, b):
    """Equal dtype, shape and bytes (torch.equal for everything but e4m3, which it does not implement)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8)) if a.dtype == torch.float8_e4m3fn else torch.equal(a, b)


def changed_elements(manifest):
    """Stored elements the merge changed against the base in the same storage (v1: BF16; v2: BF16, e4m3 against the
    base's own block-FP8 quantization, and the scales)."""
    field = 'changed_bf16_elements' if manifest['format'] == FORMAT else 'changed_elements'
    return sum(k[field] for k in manifest['keys'].values())


class MergedExporter:
    """Holds the BF16 base of the adapted projections (read once from ``model_dir``) and writes merged versions."""

    def __init__(self, model_dir, keys, rank, alpha, dense_fp8_keys=(), vision=None, full=False):
        """``vision``: (rank, alpha) of the ViT projections' LoRA (keys under VISION_KEYS), which has its own; it must
        be given exactly when such keys are. ``full``: full-weight trainer (write_full only), ``rank`` / ``alpha``
        None."""
        self.model_dir = Path(model_dir)
        self.full = full
        if full and (rank is not None or alpha is not None or vision is not None or not dense_fp8_keys):
            raise ValueError('full mode has no LoRA rank / alpha / vision LoRA and needs the dense FP8 keys (v2)')
        self.rank, self.alpha = rank, alpha
        self.scale = None if full else alpha / rank
        self.dense_fp8 = frozenset(dense_fp8_keys)
        if not self.dense_fp8 <= set(keys):
            raise KeyError(f'dense FP8 keys that are not adapted projections: {sorted(self.dense_fp8 - set(keys))[:3]}')
        self.vision_keys = frozenset(k for k in keys if k.startswith(VISION_KEYS))
        if bool(self.vision_keys) != (vision is not None):
            raise ValueError(f'{len(self.vision_keys)} ViT keys with vision LoRA settings {vision}')
        if self.vision_keys & self.dense_fp8:
            raise KeyError('ViT projections are BF16 in the deployed model, not dense FP8')
        self.vision = None if vision is None else {'rank': vision[0], 'alpha': vision[1], 'scale': vision[1] / vision[0],
                                                   'tensors': len(self.vision_keys)}
        self.format = FORMAT_V2 if self.dense_fp8 else FORMAT
        index = json.loads((self.model_dir / 'model.safetensors.index.json').read_text())['weight_map']
        missing = [k for k in keys if k not in index]
        if missing:
            raise KeyError(f'{len(missing)} adapted keys not in {self.model_dir}: {missing[:3]}')
        by_shard = defaultdict(list)
        for key in keys:
            by_shard[index[key]].append(key)
        started = time.perf_counter()
        self.base = {}
        for shard, shard_keys in sorted(by_shard.items()):
            with safe_open(str(self.model_dir / shard), framework='pt') as f:
                for key in shard_keys:
                    self.base[key] = f.get_tensor(key)
        wrong = {k: str(t.dtype) for k, t in self.base.items() if t.dtype != torch.bfloat16 or t.dim() != 2}
        if wrong:
            raise TypeError(f'adapted base weights must be 2-D BF16: {dict(list(wrong.items())[:3])}')
        self.base_sha256 = {k: tensor_sha256(t) for k, t in self.base.items()}
        self.load_seconds = time.perf_counter() - started

    def merge(self, pairs):
        if self.full:
            raise RuntimeError('a full-weight exporter has no LoRA to merge (write_full)')
        if set(pairs) != set(self.base):
            raise KeyError(f'adapter keys differ from the loaded base: {sorted(set(pairs) ^ set(self.base))[:3]}')
        merged, changed = {}, {}
        for key, (a, b) in pairs.items():
            w = self.base[key]
            rank, scale = (self.vision['rank'], self.vision['scale']) if key in self.vision_keys else (self.rank,
                                                                                                    self.scale)
            if a.dim() != 2 or b.dim() != 2 or a.shape[0] != rank or b.shape[1] != rank \
                    or (b.shape[0], a.shape[1]) != tuple(w.shape):
                raise ValueError(f'{key}: A {tuple(a.shape)} B {tuple(b.shape)} for W {tuple(w.shape)} (rank {rank})')
            if key in self.dense_fp8:
                self.merge_fp8(key, w, a, b, merged, changed)
                continue
            delta = (b.float() @ a.float()) * scale
            out = (w.float() + delta).to(torch.bfloat16)
            if not torch.isfinite(out).all():
                raise RuntimeError(f'{key}: nonfinite merged weight')
            merged[key] = out.contiguous()
            changed[key] = int((out != w).sum())
        return merged, changed

    def merge_fp8(self, key, w, a, b, merged, changed):
        """``key`` -> e4m3 and ``scale_key(key)`` -> fp32 scale_inv of the merged weight, on the current CUDA device."""
        from fp8_dense_qat import effective_weight, quantize_block_fp8

        device = torch.device('cuda', torch.cuda.current_device())
        w_eff = effective_weight(w.to(device), a.to(device), b.to(device), self.scale)
        if not torch.isfinite(w_eff).all():
            raise RuntimeError(f'{key}: nonfinite merged weight')
        q, sw = quantize_block_fp8(w_eff)
        q0, sw0 = quantize_block_fp8(w.to(device).float())
        merged[key], merged[scale_key(key)] = q.cpu().contiguous(), sw.cpu().contiguous()
        changed[key] = int((q.view(torch.uint8) != q0.view(torch.uint8)).sum())
        changed[scale_key(key)] = int((sw != sw0).sum())

    def check_against_forward(self, merged):
        """Debug (fp8_dense_qat.CAPTURE set): the tensors about to be written are those of the trainer's latest forward."""
        import fp8_dense_qat

        captured = fp8_dense_qat.CAPTURE
        for key in sorted(self.dense_fp8):
            if key not in captured:
                raise RuntimeError(f'{key}: no forward since the last export to compare the export with')
            want = captured[key]
            have = (tensor_sha256(merged[key]), tensor_sha256(merged[scale_key(key)]))
            if want != have:
                raise RuntimeError(f'{key}: exported e4m3/scale_inv differ from the trainer forward\'s ({want} vs {have})')
        captured.clear()
        return len(self.dense_fp8)

    def manifest_entry(self, key, tensor, changed):
        if self.format == FORMAT:
            return {'shape': list(tensor.shape), 'base_sha256': self.base_sha256[key],
                    'merged_sha256': tensor_sha256(tensor), 'changed_bf16_elements': changed}
        if tensor.dtype == torch.float8_e4m3fn:
            return {'shape': list(tensor.shape), 'format': FP8_FORMAT, 'scale_key': scale_key(key),
                    'base_sha256': self.base_sha256[key], 'merged_sha256': tensor_sha256(tensor),
                    'changed_elements': changed}
        if key in self.base:
            return {'shape': list(tensor.shape), 'format': 'bf16', 'base_sha256': self.base_sha256[key],
                    'merged_sha256': tensor_sha256(tensor), 'changed_elements': changed}
        return {'shape': list(tensor.shape), 'format': SCALE_FORMAT, 'merged_sha256': tensor_sha256(tensor),
                'changed_elements': changed}

    def write(self, target, pairs, step, extra=None):
        """Merge and publish version ``step`` at ``target`` (must not exist). Returns the manifest."""
        target = Path(target)
        staging = target.with_name(target.name + '.tmp')
        if target.exists() or staging.exists():
            raise FileExistsError(f'{target} or its staging directory exists')
        started = time.perf_counter()
        merged, changed = self.merge(pairs)
        forward_checked = None
        if self.dense_fp8:
            import fp8_dense_qat

            if fp8_dense_qat.CAPTURE is not None:
                forward_checked = self.check_against_forward(merged)
        return self.publish(target, merged, changed, step, extra, started, forward_checked)

    def full_weights(self, weights):
        """write_full's tensors: {key: deployed tensor(s)} and changed-element counts, every key of the set."""
        from fp8_dense_qat import quantize_block_fp8

        if not self.full:
            raise RuntimeError('write_full needs a full-mode exporter')
        unknown = set(weights) - set(self.base)
        if unknown:
            raise KeyError(f'weights outside the exported set: {sorted(unknown)[:3]}')
        device = torch.device('cuda', torch.cuda.current_device())
        merged, changed = {}, {}
        for key, base in self.base.items():
            w = weights.get(key, base)
            if w.dtype != torch.bfloat16 or w.shape != base.shape:
                raise TypeError(f'{key}: {w.dtype} {tuple(w.shape)}, the base is BF16 {tuple(base.shape)}')
            if not torch.isfinite(w).all():
                raise RuntimeError(f'{key}: nonfinite weight')
            if key in self.dense_fp8:
                q, sw = quantize_block_fp8(w.to(device).float())
                q0, sw0 = quantize_block_fp8(base.to(device).float())
                merged[key], merged[scale_key(key)] = q.cpu().contiguous(), sw.cpu().contiguous()
                changed[key] = int((q.view(torch.uint8) != q0.view(torch.uint8)).sum())
                changed[scale_key(key)] = int((sw != sw0).sum())
            else:
                merged[key] = w.contiguous()
                changed[key] = int((w != base).sum())
        return merged, changed

    def write_full(self, target, weights, step, extra=None):
        """Publish version ``step`` of a full-weight trainer at ``target`` (must not exist): ``weights`` {key: BF16 CPU
        tensor} of the trained keys, the base for the others. Returns the manifest."""
        target = Path(target)
        staging = target.with_name(target.name + '.tmp')
        if target.exists() or staging.exists():
            raise FileExistsError(f'{target} or its staging directory exists')
        started = time.perf_counter()
        merged, changed = self.full_weights(weights)
        return self.publish(target, merged, changed, step,
                            {**(extra or {}), 'trainer': 'full', 'trained_keys': len(weights)}, started, None)

    def publish(self, target, merged, changed, step, extra, started, forward_checked):
        staging = target.with_name(target.name + '.tmp')
        staging.mkdir(parents=True)
        path = staging / 'model.safetensors'
        save_file(merged, str(path), metadata={'format': 'pt', 'fn_lora_merged': self.format, 'weight_version': str(step)})
        back = load_file(str(path))
        if set(back) != set(merged) or any(not same_tensor(back[k], merged[k]) for k in merged):
            raise RuntimeError(f'{path}: read-back differs from the merged tensors')
        manifest = {
            'format': self.format, 'weight_version': str(step), 'step': step, 'base_model': str(self.model_dir),
            'lora_rank': self.rank, 'lora_alpha': self.alpha, 'scale': self.scale, 'tensors': len(merged),
            'bytes': path.stat().st_size, 'seconds': time.perf_counter() - started, **(extra or {}),
            **({'dense_qat': {'scheme': 'block_fp8_e4m3_128x128', 'scale_suffix': 'weight_scale_inv',
                              'modules': len(self.dense_fp8), 'forward_checked': forward_checked}}
               if self.dense_fp8 else {}),
            **({'vision_lora': self.vision} if self.vision else {}),
            'keys': {k: self.manifest_entry(k, t, changed[k]) for k, t in sorted(merged.items())}}
        (staging / 'merged_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
        staging.rename(target)
        (target / 'DONE').write_text('')
        return manifest
