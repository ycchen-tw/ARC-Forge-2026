"""Stream official native MTP tensors into N independent BF16 layers."""

import argparse
import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path

import torch
from attention_sink import keep_sinks_float32
from model import MTP3, Config
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Source:
    def __init__(self, path):
        self.path = Path(path)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self.audit = {}

    def read(self, key):
        with safe_open(str(self.path / self.index[key]), framework="pt", device="cpu") as f:
            value = f.get_tensor(key)
        record = {"shape": list(value.shape), "dtype": str(value.dtype), "source_sha256": digest_tensor(value)}
        if value.dtype == torch.float8_e4m3fn:
            scale_key = key.removesuffix(".weight") + ".weight_scale_inv"
            scale = self.read(scale_key)
            if value.ndim != 2 or list(scale.shape) != [(value.shape[0] + 127) // 128, (value.shape[1] + 127) // 128]:
                raise ValueError(f"unsupported FP8 layout: {key}")
            value = (
                value.float()
                * scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[: value.shape[0], : value.shape[1]]
            ).bfloat16()
            record["conversion"] = "FP8 E4M3 * BF16 block128 scale in FP32, rounded once to BF16"
        elif value.dtype != torch.bfloat16:
            raise ValueError(f"unexpected source dtype {key}: {value.dtype}")
        record["bf16_sha256"] = digest_tensor(value)
        self.audit[key] = record
        return value


def config_from_source(path, num_layers):
    text = json.loads((Path(path) / "config.json").read_text())["text_config"]
    values = {f.name: text[f.name] for f in fields(Config) if f.name in text}
    values["rope_theta"] = text["mtp"]["rope_theta"]
    # The source has exactly one MTP layer; the draft depth is ours to choose,
    # so it never comes from the official config.
    values["num_layers"] = num_layers
    c = Config(**values)
    if (c.hidden_size, c.hc_count, c.num_experts, text["mtp_num_hidden_layers"]) != (2560, 4, 512, 1):
        raise ValueError("requires full, unpruned Qwen3.8-Flash-Next native MTP")
    return c


@torch.no_grad()
def initialize(source, c):
    with torch.device("meta"):
        model = MTP3(c).to(dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    copied = []
    for name, target in model.named_parameters():
        if name.startswith("layers."):
            layer = int(name.split(".")[1])
            suffix = name.split(".", 2)[2]
            if "conv." in suffix:
                continue
            if layer > 0:
                original = model.layers[0].get_parameter(suffix)
                target.copy_(original)
                if target.data_ptr() == original.data_ptr() or not torch.equal(target, original):
                    raise AssertionError(f"layer storage/value audit failed: {name}")
                copied.append(name)
                continue
            if suffix.startswith("mlp.experts."):
                for e in range(c.num_experts):
                    if suffix.endswith("gate_up_proj"):
                        gate = source.read(f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight")
                        up = source.read(f"mtp.layers.0.mlp.experts.{e}.up_proj.weight")
                        value = torch.cat((gate, up))
                    else:
                        value = source.read(f"mtp.layers.0.mlp.experts.{e}.down_proj.weight")
                    target[e].copy_(value)
                    if not torch.equal(target[e], value):
                        raise AssertionError(name)
                copied.append(name)
                continue
            key = "mtp.layers.0." + suffix
        elif name.startswith("candidate_selector.") or name == "mask_embedding":
            continue
        elif name.startswith("target_reducer."):
            key = name.replace("target_reducer.", "model.language_model.hyper_connection_mixer.")
        elif name == "embed_tokens.weight":
            key = "model.language_model.embed_tokens.weight"
        elif name == "lm_head.weight":
            key = name
        else:
            key = "mtp." + name
        value = source.read(key)
        target.copy_(value)
        if not torch.equal(target, value):
            raise AssertionError(f"source tensor equality failed: {name}")
        copied.append(name)
    for layer in model.layers:
        for conv in (layer.attention_conv, layer.mlp_conv):
            conv.base_kernel.zero_()
            conv.base_kernel[:, 0] = 1
            conv.kernel_projection.weight.zero_()
    selector = model.candidate_selector
    torch.nn.init.normal_(selector.predecessor_codebook, std=0.02)
    selector.successor_codebook.zero_()
    selector.hidden_projection.reset_parameters()
    # A trainable private mask vector; the target vocabulary is never extended.
    model.mask_embedding.copy_(model.embed_tokens.weight[248077])
    return model, copied


def load_checkpoint(path, *, dtype=torch.bfloat16, device="cpu"):
    path = Path(path)
    c = Config(**json.loads((path / "config.json").read_text()))
    manifest = json.loads((path / "initialization_manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("incomplete initialization")
    with torch.device("meta"):
        model = MTP3(c).to(dtype=dtype)
    model.to_empty(device=device)
    keep_sinks_float32(model)
    loaded = set()
    for filename in manifest["files"]:
        state = load_file(str(path / filename), device=str(device))
        with torch.no_grad():
            for key, value in state.items():
                model.get_parameter(key).copy_(value)
                loaded.add(key)
        del state  # Release this layer before loading the next large file.
    if loaded != set(dict(model.named_parameters())):
        raise ValueError("checkpoint parameter coverage mismatch")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    # Required, not defaulted: draft depth sets the checkpoint size, the trainer
    # memory ceiling and the serving draft cost. Inheriting it silently would
    # let a run spend days at the wrong depth before anyone noticed.
    p.add_argument("--layers", required=True, type=int, help="independent draft decoder layers")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(20260909)
    c = config_from_source(args.source, args.layers)
    source = Source(args.source)
    model, copied = initialize(source, c)
    files = []
    state = model.state_dict()
    for i in range(c.num_layers):
        filename = f"layer-{i}.safetensors"
        save_file({k: v for k, v in state.items() if k.startswith(f"layers.{i}.")}, str(args.out / filename))
        files.append(filename)
    save_file({k: v for k, v in state.items() if not k.startswith("layers.")}, str(args.out / "interface.safetensors"))
    files.append("interface.safetensors")
    manifest = {
        "complete": True,
        "source": str(args.source.resolve()),
        "source_config_sha256": hashlib.sha256((args.source / "config.json").read_bytes()).hexdigest(),
        "source_index_sha256": hashlib.sha256((args.source / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "source_tensors": source.audit,
        "copied_parameters": copied,
        "files": files,
        "layer_storage_independent": True,
        "copied_layers_bitwise_equal": True,
        "comparison_domain": "BF16 dequantization of the official FP8 checkpoint",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "attention": "causal SWA: 0 <= query_position - key_position < 1024; no QSA indexer",
        "mask_token_embedding_source_id": 248077,
        "extra_final_norm": False,
        "production_gates_passed": False,
    }
    (args.out / "config.json").write_text(json.dumps(asdict(c), indent=2) + "\n")
    (args.out / "initialization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("source_tensors", "copied_parameters")}, indent=2))


if __name__ == "__main__":
    main()
