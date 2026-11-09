"""Block-FP8 the non-MoE BF16 linears of the Qwen3.8-FN MXFP4 checkpoint.

The published checkpoint quantizes routed experts only: 2,198 of its 3,097
preserved tensors are still BF16 (10.10 GiB excluding the PLE table). This
converts the safely-quantizable dense projections to the same block-FP8 format
the config already declares (``weight_block_size: [128, 128]``), which SGLang's
``Fp8LinearMethod`` consumes as ``weight`` (fp8_e4m3) + ``weight_scale_inv``
(fp32, ``[ceil(out/128), ceil(in/128)]``).

Excluded on purpose:
  * router (``mlp.gate``) and hyper-connection mixers -- routing/mixing are
    precision-sensitive, and the hyper-connection shapes (320) are not a
    multiple of the 128 block anyway.
  * ``embed_tokens`` / ``lm_head`` -- lm_head has a dedicated server flag
    (``--enable-fp8-lm-head``); the embedding is a gather, not a GEMM.
  * the vision tower -- mixed 128-divisibility and multimodal correctness risk.
  * anything whose in/out dims are not both divisible by 128: SGLang raises
    on those (fp8.py, create_weights).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BLOCK = 128
FP8_MAX = 448.0
# Suffix patterns of dense projections that are plain GEMMs in the forward pass.
INCLUDE = re.compile(
    r"\.(?:"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
    r"|self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|self_attn\.indexer\.index_qk_proj"
    r"|mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj)"
    r")\.weight$"
)
EXCLUDE = re.compile(r"\.ple\.|visual|hyper_connection|\.mlp\.gate\.weight$|embed_tokens|lm_head")


def selectable(name: str, dtype: str, shape) -> bool:
    if dtype != "BF16" or len(shape) != 2:
        return False
    if EXCLUDE.search(name) or not INCLUDE.search(name):
        return False
    return shape[0] % BLOCK == 0 and shape[1] % BLOCK == 0


def quantize(weight: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-[128,128]-block e4m3, DeepSeek-style scale_inv."""
    out_dim, in_dim = weight.shape
    w = weight.to(device=device, dtype=torch.float32)
    blocked = w.view(out_dim // BLOCK, BLOCK, in_dim // BLOCK, BLOCK)
    amax = blocked.abs().amax(dim=(1, 3))
    # A fully-zero block would divide by zero; its quantized value is zero either way.
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = blocked / scale[:, None, :, None]
    q = q.clamp(-FP8_MAX, FP8_MAX).view(out_dim, in_dim).to(torch.float8_e4m3fn)
    return q.cpu(), scale.to(torch.float32).cpu()


def rel_error(weight: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, device: str) -> float:
    out_dim, in_dim = weight.shape
    w = weight.to(device=device, dtype=torch.float32)
    deq = (q.to(device=device, dtype=torch.float32)
           .view(out_dim // BLOCK, BLOCK, in_dim // BLOCK, BLOCK)
           * scale.to(device)[:, None, :, None]).view(out_dim, in_dim)
    return float((deq - w).pow(2).sum() / w.pow(2).sum().clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu, or cuda when the GPU is free")
    parser.add_argument("--link-mode", choices=["auto", "symlink"], default="auto",
                        help="Use symlinks for unchanged shards on a read-only Kaggle mount")
    parser.add_argument("--error-sample", type=int, default=24)
    args = parser.parse_args()

    src, out = args.src.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = dict(index["weight_map"])

    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)

    started = time.time()
    report = {"converted": 0, "converted_bytes_before": 0, "converted_bytes_after": 0,
              "rewritten_shards": [], "linked_shards": [], "errors": [], "complete": False}
    converted_names: set[str] = set()

    for shard in sorted(shards):
        path = src / shard
        with safe_open(path, framework="pt") as reader:
            keys = list(reader.keys())
            targets = set()
            for k in keys:
                sl = reader.get_slice(k)
                if selectable(k, sl.get_dtype(), tuple(sl.get_shape())):
                    targets.add(k)
        if not targets:
            dest = out / shard
            if not dest.exists():
                if args.link_mode == "symlink":
                    dest.symlink_to(path.resolve())
                elif path.stat().st_dev == out.stat().st_dev:
                    os.link(path, dest)
                else:
                    shutil.copy2(path, dest)
            report["linked_shards"].append(shard)
            continue

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt") as reader:
            for key in keys:
                tensor = reader.get_tensor(key)
                if key in targets:
                    q, scale = quantize(tensor, args.device)
                    if len(report["errors"]) < args.error_sample:
                        report["errors"].append(
                            {"name": key, "rel_mse": rel_error(tensor, q, scale, args.device)})
                    tensors[key] = q
                    tensors[key[: -len("weight")] + "weight_scale_inv"] = scale
                    report["converted"] += 1
                    report["converted_bytes_before"] += tensor.numel() * 2
                    report["converted_bytes_after"] += q.numel() + scale.numel() * 4
                    converted_names.add(key)
                else:
                    tensors[key] = tensor
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        report["rewritten_shards"].append(shard)
        print(f"rewrote {shard}: {len(targets)} converted", flush=True)

    # index: add the new scale entries next to their weight
    for name in sorted(converted_names):
        weight_map[name[: -len("weight")] + "weight_scale_inv"] = weight_map[name]
    index["weight_map"] = weight_map
    index.setdefault("metadata", {})["total_size"] = sum(
        (out / shard).stat().st_size for shard in shards)
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # config: stop excluding the modules we just quantized
    config = json.loads((src / "config.json").read_text())
    quant = config["quantization_config"]
    skip = set(quant.get("modules_to_not_convert") or [])
    removed = sorted(n[: -len(".weight")] for n in converted_names
                     if n[: -len(".weight")] in skip)
    quant["modules_to_not_convert"] = sorted(skip - set(removed))
    quant["nonmoe_fp8"] = {"block": [BLOCK, BLOCK], "converted": report["converted"],
                           "method": "rtn-symmetric-per-block"}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    report["modules_unskipped"] = len(removed)
    report["skip_list_before"] = len(skip)
    report["skip_list_after"] = len(quant["modules_to_not_convert"])

    for extra in src.iterdir():
        if extra.is_file() and extra.name not in {"config.json", "model.safetensors.index.json"} \
                and not extra.name.endswith(".safetensors"):
            shutil.copy2(extra, out / extra.name)

    report["saved_gib"] = (report["converted_bytes_before"] - report["converted_bytes_after"]) / 2**30
    report["elapsed_s"] = time.time() - started
    report["complete"] = True
    (out / "nonmoe_fp8_manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("converted", "saved_gib", "modules_unskipped", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
