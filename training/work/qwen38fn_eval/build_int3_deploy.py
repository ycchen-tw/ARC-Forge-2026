"""Repackage local gEAN INT3 parts + existing FP8 dense into a compact HF checkpoint.

No quantization, fake-quant carrier, CUDA, or NFS reads. Main routed weights use
Humming packed uint3 g128; GDN in/out and attention qkv use existing block FP8;
attention o_proj stays BF16. Everything else (including PLE and MTP) is copied.
The opt-in arc_int3 serving plugin must consume config.quantization_config.arc_int3.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MAIN_EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.")
DENSE_FP8 = re.compile(
    r"^model\.language_model\.layers\.\d+\.(?:linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
    r"|self_attn\.(?:q_proj|k_proj|v_proj))\.weight$"
)
O_PROJ = re.compile(r"^model\.language_model\.layers\.\d+\.self_attn\.o_proj\.weight$")


def require_local(path: Path, mounts: list[tuple[str, str]] | None = None) -> Path:
    """Check mount type before even lstat; resolve symlinks one local component at a time."""
    if mounts is None:
        mounts = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            mount = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), left.split()[4])
            mounts.append((mount, right.split()[0]))
    mounts = sorted(mounts, key=lambda item: len(item[0]), reverse=True)
    pending = list(Path(os.path.abspath(path)).parts[1:])
    current, links = Path("/"), 0
    while pending:
        component = pending.pop(0)
        if component == "..":
            current = current.parent
            continue
        if component == ".":
            continue
        candidate = current / component
        match = next((fs for root, fs in mounts
                      if str(candidate) == root or str(candidate).startswith(root.rstrip("/") + "/")), None)
        # /dev is devtmpfs; it is local and only traversed on the way to /dev/shm (tmpfs).
        if match not in {"ext4", "tmpfs"} and not (match == "devtmpfs" and pending):
            raise RuntimeError(f"input/output must be local ext4 or tmpfs: {candidate} ({match})")
        if candidate.is_symlink():
            links += 1
            if links > 40:
                raise RuntimeError(f"symlink loop: {path}")
            target = Path(os.readlink(candidate))
            if target.is_absolute():
                current = Path("/")
                pending = list(target.parts[1:]) + pending
            else:
                pending = list(target.parts) + pending
        else:
            current = candidate
    return current


def digest(path: Path) -> str:
    path = require_local(path)
    h = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(16 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(require_local(path).read_text())


def json_file(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def shard_path(root: Path, name: str) -> Path:
    if Path(name).name != name or not name.endswith(".safetensors"):
        raise RuntimeError(f"index contains unsafe shard path: {name}")
    return require_local(root / name)


def load_tensor(root: Path, index: dict, name: str) -> torch.Tensor:
    with safe_open(shard_path(root, index[name]), framework="pt", device="cpu") as reader:
        return reader.get_tensor(name)


def source_identity(root: Path) -> dict:
    return {"path": str(root), "config_sha256": digest(root / "config.json"),
            "index_sha256": digest(root / "model.safetensors.index.json")}


class Shards:
    """Bound memory by a shard budget plus the current tensor; no complete-model buffer."""
    def __init__(self, out: Path, budget: int):
        if budget <= 0:
            raise ValueError("shard budget must be positive")
        self.out, self.budget = out, budget
        self.buf, self.size, self.total = {}, 0, 0
        self.index, self.files, self.seen = {}, {}, set()

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.seen or tensor.device.type != "cpu":
            raise RuntimeError(f"duplicate name or non-CPU tensor: {name}")
        size = tensor.numel() * tensor.element_size()
        if self.buf and self.size + size > self.budget:
            self.flush()
        self.buf[name] = tensor.contiguous()
        self.size += size
        self.total += size
        self.seen.add(name)
        if self.size >= self.budget:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        name = f"model-{len(self.files) + 1:05d}.safetensors"
        temp = self.out / (name + ".partial")
        save_file(self.buf, str(temp), metadata={"format": "pt"})
        temp.replace(self.out / name)
        self.files[name] = {"sha256": digest(self.out / name), "bytes": (self.out / name).stat().st_size}
        self.index.update({key: name for key in self.buf})
        print(f"SHARD {name} tensors={len(self.buf)} payload={self.size}", flush=True)
        self.buf, self.size = {}, 0


def validate_keep(keep: dict, layers: int, experts: int) -> dict:
    if set(keep) != {str(layer) for layer in range(layers)}:
        raise RuntimeError("keep_per_layer must cover exactly the main model layers")
    for layer, ids in keep.items():
        if len(ids) != experts or len(set(ids)) != experts or any(type(i) is not int or i < 0 for i in ids):
            raise RuntimeError(f"invalid original expert IDs at layer {layer}")
    return keep


def quant_config(config: dict, main: list[str], dense: list[str], bf16: list[str], experts: int,
                 drop_mtp: bool = False) -> dict:
    q = copy.deepcopy(config["quantization_config"])
    removed = list(main)
    if drop_mtp:
        removed += [k for k in q["quantized_layers"] if k.startswith("mtp.")]
        removed += [t for g in q["config_groups"].values() for t in g["targets"] if t.startswith("mtp.")]
    q["quant_method"] = "modelopt_mixed"
    q["quant_algo"] = "MIXED_PRECISION"
    # SGLang treats modelopt_mixed config.json without KV metadata as incomplete and
    # falls back to hf_quant_config.json, which this package deliberately omits.
    q.setdefault("kv_cache_quant_algo", None)
    q["quantized_layers"] = {k: v for k, v in q["quantized_layers"].items() if k not in removed}
    groups = {}
    for key, group in q["config_groups"].items():
        group["targets"] = [target for target in group["targets"] if target not in removed]
        if group["targets"]:
            groups[key] = group
    q["config_groups"] = groups
    q["arc_int3"] = {
        "schema_version": 1, "bits": 3, "group_size": 128, "num_experts": experts,
        "scale_dtype": "bfloat16", "layout": "humming_uint3_g128_nozp", "w13_rows": "gate_then_up",
        "dense_fp8_modules": dense, "dense_bf16_modules": bf16, "main_expert_modules": main,
    }
    return q


def model_card(manifest: dict, notes: str | None = None) -> str:
    def calibration(value):
        if value is None:
            return "Pending: the actual calibration manifest has not been attached."
        selected = {key: value[key] for key in (
            "method", "formula", "sources", "arc2_sources", "pools", "tokens", "token_count", "total_tokens",
            "images", "videos", "media_mode", "arc3_public25_split", "arc3_public25_games", "arc3_public25_tainted",
            "source_commit", "mask_sha256", "manifest_sha256"
        ) if key in value}
        return ("`" + json.dumps(selected, ensure_ascii=False) + "`" if selected else
                "Recorded in deploy_manifest.json; inspect its calibration identity.")

    return f"""---
tags: [sglang, moe, int3, gptq, arc-agi]
---
# Packed gEAN{manifest['main_experts']} INT3 GPTQ

This is a compact deployment checkpoint, with packed weights rather than BF16 fake-quant tensors.
The exact per-layer original expert IDs are recorded in [deploy_manifest.json](deploy_manifest.json).
Keep-list SHA256: `{manifest['keep_per_layer_sha256']}`. GPTQ run: `{manifest['sidecar_run_id']}`.

## Calibration and evaluation boundaries

Expert selection (pruning): {calibration(manifest['pruning_calibration'])}
INT3 GPTQ calibration: {calibration(manifest['gptq_calibration'])}
These are separate data uses. If the pruning record includes ARC3 public-25 A (12 games),
public-25 is not an untouched test set; report that overlap with every comparison.
No calibration counts, source hashes or task scores are inferred from older runs.

## Tensor formats and runtime

Main routed experts: Humming packed uint3 g128, no zero point, signed BF16 scales;
dequantization is `(code - 4) * scale`, w13 rows are gate then up.
GDN in/out and attention q/k/v are existing block-FP8 weights; attention o_proj stays BF16.
Other tensors are copied unchanged from the pruned source, including PLE, vision, shared experts,
routers, embeddings, lm_head{'' if manifest.get('mtp_included', True) else ' (no MTP: its tensors are omitted and mtp_num_hidden_layers is 0; pair with an external drafter such as DFlash)'}{' and MTP' if manifest.get('mtp_included', True) else ''}. PLE remains in its original format; deploy with PLE offloaded to host.

Serving requires the opt-in arc_int3 SGLang plugin and a compatible Humming runtime.
The config uses modelopt_mixed with explicit arc_int3 overrides; a stock HF/SGLang loader is insufficient.
Runtime/source revisions: pending until the G4 deployment receipt records the actual versions.

## Size and validation

Tensor payload: {manifest['payload_bytes']:,} bytes; file sizes and SHA256 are in the deployment manifest.
The older 32.2 GB sidecar figure is not this complete model's size or VRAM requirement.
Colab G4 SM120 kernel, load, memory, throughput and ARC3 validation: **pending**.
Compare this release against old REAP320 NVFP4 as complete deployments: mask and dense precision
also differ, so the result does not isolate the causal effect of the INT3 format.
""" + ("" if notes is None else "\n" + notes.strip() + "\n")


def build(*, sidecar: Path, source: Path, dense_fp8: Path, dense_manifest: Path,
          out: Path, bf16: Path | None = None, shard_bytes: int = 1024**3,
          expected_experts: int = 320, gptq_manifest: Path | None = None,
          pruning_calibration_manifest: Path | None = None, validation_notes: Path | None = None,
          drop_mtp: bool = False) -> dict:
    # Validate filesystem before any reads/resolve that could touch a hard NFS mount.
    sidecar, source, dense_fp8, dense_manifest, out = [require_local(p) for p in
                                                     (sidecar, source, dense_fp8, dense_manifest, out)]
    if bf16 is not None:
        bf16 = require_local(bf16)
    gptq_meta = None if gptq_manifest is None else read_json(require_local(gptq_manifest))
    prune_meta = (None if pruning_calibration_manifest is None else
                  read_json(require_local(pruning_calibration_manifest)))
    if out.exists():
        raise RuntimeError(f"output already exists: {out}")
    config = read_json(source / "config.json")
    text = config["text_config"]
    experts, layers = text["num_experts"], text["num_hidden_layers"]
    hidden, inter = text["hidden_size"], text["moe_intermediate_size"]
    if experts != expected_experts or hidden % 128 or inter % 128:
        raise RuntimeError("unexpected expert count or non-g128 geometry")
    prune = read_json(source / "prune_manifest.json")
    keep = validate_keep(prune["keep_per_layer"], layers, experts)
    sm = read_json(sidecar / "int3_manifest.json")
    if validate_keep(sm["keep_per_layer"], layers, experts) != keep:
        raise RuntimeError("INT3 original expert IDs differ from the gEAN source mask")
    dm = read_json(dense_manifest)
    if digest(dense_fp8) != dm["files"][dense_fp8.name]:
        raise RuntimeError("dense FP8 sha256 differs from its manifest")
    index = read_json(source / "model.safetensors.index.json")["weight_map"]
    dense_weights = sorted(name for name in index if DENSE_FP8.match(name))
    o_weights = sorted(name for name in index if O_PROJ.match(name))
    if not dense_weights or not o_weights:
        raise RuntimeError("checkpoint lacks the requested FP8/BF16 dense projections")
    main = [f"model.language_model.layers.{layer}.mlp.experts" for layer in range(layers)]
    required = set()
    # Preflight exact format and expert-order identity before creating the output.
    side_paths = []
    for layer in range(layers):
        name = f"experts-L{layer:02d}.safetensors"
        path = require_local(sidecar / name)
        if digest(path) != sm["files"][name]:
            raise RuntimeError(f"INT3 sha256 mismatch: {name}")
        prefix = main[layer]
        shapes = {"w13_qweight": (torch.int32, [experts, 2 * inter, hidden * 3 // 32]),
                  "w13_scales": (torch.bfloat16, [experts, 2 * inter, hidden // 128]),
                  "w2_qweight": (torch.int32, [experts, hidden, inter * 3 // 32]),
                  "w2_scales": (torch.bfloat16, [experts, hidden, inter // 128])}
        with safe_open(path, framework="pt", device="cpu") as reader:
            names = {f"{prefix}.{suffix}" for suffix in shapes}
            if set(reader.keys()) != names:
                raise RuntimeError(f"INT3 layer {layer} contains missing or extra tensors")
            for suffix, (dtype, shape) in shapes.items():
                tensor = reader.get_tensor(f"{prefix}.{suffix}")
                if tensor.dtype != dtype or list(tensor.shape) != shape:
                    raise RuntimeError(f"INT3 shape/dtype mismatch: {prefix}.{suffix}")
                if suffix.endswith("scales") and (not torch.isfinite(tensor).all() or (tensor == 0).any()):
                    raise RuntimeError(f"INT3 scale is zero or nonfinite: {prefix}.{suffix}")
            required.update(names)
        side_paths.append(path)
    replaced = set()
    for name in dense_weights + o_weights:
        prefix = name[:-len("weight")]
        replaced.update(key for key in index if key.startswith(prefix))
    preserved = {key for key in index if not MAIN_EXPERT.match(key) and key not in replaced
                 and not (drop_mtp and key.startswith("mtp."))}
    if drop_mtp and not any(key.startswith("mtp.") for key in index):
        raise RuntimeError("--drop-mtp given but the source has no mtp.* tensors")
    required.update(preserved)
    required.update(dense_weights + o_weights)
    required.update(name[:-len("weight")] + "weight_scale_inv" for name in dense_weights)
    out.mkdir(parents=True)
    writer = Shards(out, shard_bytes)
    source_shards = {}
    for key in sorted(preserved):
        source_shards.setdefault(index[key], []).append(key)
    source_hashes = {}
    for name, keys in source_shards.items():
        path = shard_path(source, name)
        source_hashes[name] = digest(path)
        with safe_open(path, framework="pt", device="cpu") as reader:
            for key in keys:
                writer.add(key, reader.get_tensor(key))
    for path in side_paths:
        with safe_open(path, framework="pt", device="cpu") as reader:
            for key in sorted(reader.keys()):
                writer.add(key, reader.get_tensor(key))
    with safe_open(dense_fp8, framework="pt", device="cpu") as reader:
        for name in dense_weights:
            weight = reader.get_tensor(name)
            scale_name = name[:-len("weight")] + "weight_scale_inv"
            scale = reader.get_tensor(scale_name)
            original = load_tensor(source, index, name)
            if (weight.dtype != torch.float8_e4m3fn or weight.shape != original.shape
                    or scale.dtype != torch.float32
                    or list(scale.shape) != [(weight.shape[0] + 127) // 128, (weight.shape[1] + 127) // 128]
                    or not torch.isfinite(scale).all() or (scale <= 0).any()):
                raise RuntimeError(f"existing dense FP8 format mismatch: {name}")
            del original
            writer.add(name, weight)
            writer.add(scale_name, scale)
    fallback = None if bf16 is None else read_json(bf16 / "model.safetensors.index.json")["weight_map"]
    bf16_hashes = {}
    for name in o_weights:
        weight = load_tensor(source, index, name)
        source_shape = weight.shape
        if weight.dtype == torch.uint8:
            if name[:-len("weight")] + "weight_scale_2" not in index:
                raise RuntimeError(f"unknown packed o_proj format: {name}")
            source_shape = (source_shape[0], source_shape[1] * 2)
        source_name = index[name]
        if source_name not in source_hashes:
            source_hashes[source_name] = digest(shard_path(source, source_name))
        if weight.dtype != torch.bfloat16:
            if fallback is None:
                raise RuntimeError(f"{name} is not BF16; provide the official --bf16 source")
            weight = load_tensor(bf16, fallback, name)
            filename = fallback[name]
            if filename not in bf16_hashes:
                bf16_hashes[filename] = digest(shard_path(bf16, filename))
        if weight.dtype != torch.bfloat16 or weight.shape != source_shape:
            raise RuntimeError(f"official o_proj is not BF16: {name}")
        writer.add(name, weight)
    writer.flush()
    if set(writer.index) != required:
        raise RuntimeError(f"output tensor coverage differs: missing={required - set(writer.index)}")
    config["quantization_config"] = quant_config(config, main, [n[:-7] for n in dense_weights],
                                               [n[:-7] for n in o_weights], experts, drop_mtp)
    if drop_mtp:
        config["text_config"]["mtp_num_hidden_layers"] = 0
    json_file(out / "model.safetensors.index.json", {"metadata": {"total_size": writer.total},
                                                    "weight_map": writer.index})
    json_file(out / "config.json", config)
    json_file(out / "prune_manifest.json", prune)
    shutil.copyfile(sidecar / "int3_manifest.json", out / "source_int3_manifest.json")
    # Do not ship old hf_quant_config.json: it can override the custom mixed config.
    assets = {"generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
              "chat_template.jinja", "merges.txt", "vocab.json", "tokenizer.model", "LICENSE"}
    for path in source.iterdir():
        if path.name in assets or path.suffix == ".py":
            path = require_local(path)
            if path.is_file():
                shutil.copyfile(path, out / path.name)
    files = dict(writer.files)
    for path in out.iterdir():
        if path.name not in files:
            files[path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    manifest = {
        "schema": "arcforge.qwen38fn.int3_deploy.v1", "complete": True,
        "main_experts": experts, "n_layers": layers, "keep_per_layer": keep,
        "keep_per_layer_sha256": hashlib.sha256(json.dumps(keep, sort_keys=True,
                                                           separators=(",", ":")).encode()).hexdigest(),
        "source": source_identity(source), "source_prune_manifest_sha256": digest(source / "prune_manifest.json"),
        "source_shards_sha256": source_hashes,
        "sidecar_manifest_sha256": digest(sidecar / "int3_manifest.json"), "sidecar_run_id": sm["run_id"],
        "dense_manifest_sha256": digest(dense_manifest), "dense_file_sha256": digest(dense_fp8),
        "gptq_calibration": gptq_meta if gptq_meta is not None else sm.get("gptq_calibration"),
        "pruning_calibration": prune_meta if prune_meta is not None else sm.get("pruning_calibration"),
        "gptq_manifest_sha256": None if gptq_manifest is None else digest(gptq_manifest),
        "pruning_calibration_manifest_sha256": (None if pruning_calibration_manifest is None else
                                                 digest(pruning_calibration_manifest)),
        "official_bf16": None if bf16 is None else source_identity(bf16),
        "official_bf16_shards_sha256": bf16_hashes,
        "policy": {"main_routed": "packed uint3 g128; signed BF16 scales; no zero point",
                   "dense_fp8": [n[:-7] for n in dense_weights], "dense_bf16": [n[:-7] for n in o_weights],
                   "other_tensors": "copied unchanged from the pruned NVFP4 source"
                                    + ("; mtp.* omitted" if drop_mtp else "")},
        "mtp_included": not drop_mtp,
        "payload_bytes": writer.total, "tensor_count": len(writer.index), "files": files,
    }
    notes = None if validation_notes is None else require_local(validation_notes).read_text()
    (out / "README.md").write_text(model_card(manifest, notes))
    manifest["files"]["README.md"] = {"sha256": digest(out / "README.md"),
                                        "bytes": (out / "README.md").stat().st_size}
    json_file(out / "deploy_manifest.json", manifest)  # completion receipt is written last
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sidecar", "source", "dense-fp8", "dense-manifest", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--bf16", type=Path)
    parser.add_argument("--gptq-manifest", type=Path)
    parser.add_argument("--pruning-calibration-manifest", type=Path)
    parser.add_argument("--validation-notes", type=Path, help="Markdown appended to README (measured results only)")
    parser.add_argument("--drop-mtp", action="store_true", help="omit mtp.* tensors; serve with an external drafter")
    parser.add_argument("--shard-bytes", type=int, default=1024**3)
    args = parser.parse_args()
    result = build(**vars(args))
    print(json.dumps({key: result[key] for key in ("complete", "payload_bytes", "tensor_count")}))


if __name__ == "__main__":
    main()
