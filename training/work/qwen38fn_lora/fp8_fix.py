"""Work around Transformers 5.16.1 Qwen4Exp FP8 dequantization omissions."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

_EXPERT_SCALE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight_scale_inv$"
)


def load_expert_scales(model_path: str | Path):
    """Load the small scale grids without materializing any expert FP8 weight."""
    from safetensors import safe_open

    root = Path(model_path)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    by_file = defaultdict(list)
    for key, filename in weight_map.items():
        if _EXPERT_SCALE.fullmatch(key):
            by_file[filename].append(key)
    scales = {}
    for filename, keys in by_file.items():
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                match = _EXPERT_SCALE.fullmatch(key)
                layer, expert = int(match.group(1)), int(match.group(2))
                scales[layer, expert, match.group(3)] = handle.get_tensor(key)
    return scales


def apply_expert_scales(torch, model, model_path: str | Path) -> dict[str, object]:
    """Turn raw-FP8-as-BF16 packed experts into correctly dequantized BF16."""
    scales = load_expert_scales(model_path)
    layers = model.model.layers
    expected = len(layers) * model.config.num_experts * 3
    if len(scales) != expected:
        raise RuntimeError(f"expert scale coverage {len(scales)} != expected {expected}")
    with torch.no_grad():
        for layer_index, layer in enumerate(layers):
            experts = layer.mlp.experts
            gate = torch.stack(
                [scales[layer_index, expert, "gate_proj"] for expert in range(model.config.num_experts)]
            )
            up = torch.stack(
                [scales[layer_index, expert, "up_proj"] for expert in range(model.config.num_experts)]
            )
            down = torch.stack(
                [scales[layer_index, expert, "down_proj"] for expert in range(model.config.num_experts)]
            )
            gate_up_scale = torch.cat((gate, up), dim=1).to(
                device=experts.gate_up_proj.device, dtype=experts.gate_up_proj.dtype
            )
            down_scale = down.to(device=experts.down_proj.device, dtype=experts.down_proj.dtype)
            experts.gate_up_proj.view(
                model.config.num_experts, gate_up_scale.shape[1], 128,
                gate_up_scale.shape[2], 128,
            ).mul_(gate_up_scale[:, :, None, :, None])
            experts.down_proj.view(
                model.config.num_experts, down_scale.shape[1], 128,
                down_scale.shape[2], 128,
            ).mul_(down_scale[:, :, None, :, None])
    model._arc_expert_scales_applied = len(scales)
    return {"scale_tensors": len(scales), "layers": len(layers), "experts_per_layer": model.config.num_experts}


def apply_ple_scale(torch, model, model_path: str | Path) -> float:
    from safetensors import safe_open

    root = Path(model_path)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
    with safe_open(root / weight_map[key], framework="pt", device="cpu") as handle:
        scale = float(handle.get_tensor(key).float().item())
    weight = model.model.layers[1].ple.ple_embedding.ngram_embedding.weight
    with torch.no_grad():
        weight.mul_(scale)
    model._arc_ple_scale = scale
    return scale
